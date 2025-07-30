# E-COMMERCE CUSTOMER CHURN ANALYSIS AND PREDICTION
This group final project presents a machine learning–based churn prediction model for an e-commerce platform, leveraging features from five behavioral domains: demographics, engagement, transactions, platform preference, and satisfaction. Multiple classification models were benchmarked and optimized through hyperparameter tuning, with the best model selected and evaluated using test set metrics and feature importance analysis. The project uncovers patterns behind customer disengagement and identifies high-risk customers, enabling earlier churn detection, data-driven decisions, better retention planning, and more efficient marketing spend. The final model provides an interpretable, data-driven tool to support targeted retention strategies and improve business outcomes.

---

## I - BUSINESS UNDERSTANDING

### **Context**  
Customer churn—when users stop using a service—is a major challenge in e-commerce, leading to revenue loss, higher acquisition costs, and reduced customer lifetime value. Retaining customers is significantly more cost-effective than acquiring new ones. This project uses historical data and machine learning to predict churn risk and help stakeholders implement targeted retention strategies.

### **Problem Statement**  
The company faces rising customer inactivity, threatening revenue and marketing efficiency. Key questions include identifying who will churn, understanding predictive behaviors, and detecting churn risk early. Without predictive insights, marketing budgets may be wasted and valuable customers lost. The problem is framed as a binary classification: churn (`1`) or not (`0`).

### **Goals**  
- Identify customers likely to churn early for timely intervention.  
- Provide data-driven insights to support stakeholder decisions.  
- Lower churn rates through improved retention strategies.  
- Focus marketing efforts on high-risk customer segments to optimize budget.

### **Analytical Approach**  
Develop a supervised machine learning classifier using customer demographics, behavior, and transactions. Prioritize minimizing false negatives (missed churners) with metrics like F2-score, Recall, and PR-AUC. Deliver interpretable, actionable outputs to support retention efforts.

### **Scope and Limitations**  
Scope includes modeling churn from structured historical data, analyzing key churn drivers, and providing retention insights. Limitations involve missing external factors, unclear churn definitions and timing, assumptions on feature units, class imbalance (~16.8% churn), and a predictive—not explanatory—model. Despite these, the model offers a solid foundation for churn risk assessment and retention strategy support.

---

## II - DATA EXPLORATION AND PREPARATION
This phase covers data understanding, data cleaning, and exploratory data analysis (EDA) conducted for both business and machine learning purposes, as well as data preprocessing to prepare the dataset for modeling. The dataset contains 5,630 rows and 20 columns, including 19 features related to customer demographics, behavior, and transactions, plus the target variable **Churn**.

The data cleaning process includes handling missing values, converting incorrect data types, removing duplicate data, resolving inconsistent data, and addressing outliers based on domain knowledge.

During data preprocessing, the following steps were applied:

- **Scaling (Robust Scaler):** Applied to numeric features such as `Tenure`, `WarehouseToHome`, `HourSpendOnApp`, `OrderAmountHikeFromlastYear`, `NumberOfDeviceRegistered`, `NumberOfAddress`, `CouponUsed`, `OrderCount`, `DaySinceLastOrder`, and `CashbackAmount`.

- **Encoding:**
  - One-hot encoding for nominal categorical features: `PreferredLoginDevice`, `PreferredPaymentMode`, `Gender`, `MaritalStatus`, and `PreferedOrderCat`.
  - Ordinal encoding for ordered categorical features: `CityTier` and `SatisfactionScore`.
 
- **Passthrough** (no transformation): `CustomerID` and `Complain`.

- **Feature Engineering and Binning:**
  - Created new behavioral features including `RecencyRatio`, `IsActiveUser`, and `UnhappyCustomer`.
  - Binned `Tenure` into four ordered categories:  
    - New (0–3 months)  
    - Early (3–9 months)  
    - MidTerm (9–15 months)  
    - LongTerm (>15 months)  
  with ordinal encoding applied to preserve order.

---

## III - MODELING AND EVALUATION

---

## IV - CONCLUSION AND RECOMMENDATION

---

## REFERENCES

---

## TOOLS USED

- **Python:** Main programming language for data cleaning, preprocessing, modeling, and evaluation.  
- **Pandas and NumPy:** Data manipulation and numerical operations.  
- **Scikit-learn:** Model benchmarking, hyperparameter tuning, and evaluation metrics.  
- **Matplotlib and Seaborn:** Data visualization and diagnostic plots (residuals, feature importance).  
- **SHAP:** Model interpretability and feature impact analysis.  
- **Jupyter Notebook:** Interactive development and documentation environment. 
- **Tableau:** Data exploration, dashboarding, and visual analytics.  
- **Streamlit:** Web app framework for deploying interactive machine learning models.

