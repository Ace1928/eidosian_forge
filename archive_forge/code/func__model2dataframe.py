from patsy import dmatrix
import pandas as pd
from statsmodels.api import OLS
from statsmodels.api import stats
import numpy as np
import logging
def _model2dataframe(model_endog, model_exog, model_type=OLS, **kwargs):
    """return a series containing the summary of a linear model

    All the exceding parameters will be redirected to the linear model
    """
    model_result = model_type(model_endog, model_exog, **kwargs).fit()
    statistics = pd.Series({'r2': model_result.rsquared, 'adj_r2': model_result.rsquared_adj})
    result_df = pd.DataFrame({'params': model_result.params, 'pvals': model_result.pvalues, 'std': model_result.bse, 'statistics': statistics})
    fisher_df = pd.DataFrame({'params': {'_f_test': model_result.fvalue}, 'pvals': {'_f_test': model_result.f_pvalue}})
    res_series = pd.concat([result_df, fisher_df]).unstack()
    return res_series.dropna()