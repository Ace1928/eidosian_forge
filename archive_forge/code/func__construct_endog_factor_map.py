from statsmodels.compat.pandas import MONTH_END, QUARTER_END
from collections import OrderedDict
from warnings import warn
import numpy as np
import pandas as pd
from scipy.linalg import cho_factor, cho_solve, LinAlgError
from statsmodels.tools.data import _is_using_pandas
from statsmodels.tools.validation import int_like
from statsmodels.tools.decorators import cache_readonly
from statsmodels.regression.linear_model import OLS
from statsmodels.genmod.generalized_linear_model import GLM
from statsmodels.multivariate.pca import PCA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.statespace._quarterly_ar1 import QuarterlyAR1
from statsmodels.tsa.vector_ar.var_model import VAR
from statsmodels.tools.tools import Bunch
from statsmodels.tools.validation import string_like
from statsmodels.tsa.tsatools import lagmat
from statsmodels.tsa.statespace import mlemodel, initialization
from statsmodels.tsa.statespace.tools import (
from statsmodels.tsa.statespace.kalman_smoother import (
from statsmodels.base.data import PandasData
from statsmodels.iolib.table import SimpleTable
from statsmodels.iolib.summary import Summary
from statsmodels.iolib.tableformatting import fmt_params
def _construct_endog_factor_map(self, factors, endog_names):
    """
        Construct mapping of observed variables to factors.

        Parameters
        ----------
        factors : dict
            Dictionary of {endog_name: list of factor names}
        endog_names : list of str
            List of the names of the observed variables.

        Returns
        -------
        endog_factor_map : pd.DataFrame
            Boolean dataframe with `endog_names` as the index and the factor
            names (computed from the `factors` input) as the columns. Each cell
            is True if the associated factor is allowed to load on the
            associated observed variable.

        """
    missing = []
    for key, value in factors.items():
        if not isinstance(value, (list, tuple)) or len(value) == 0:
            missing.append(key)
    if len(missing):
        raise ValueError(f'Each observed variable must be mapped to at least one factor in the `factors` dictionary. Variables missing factors are: {missing}.')
    missing = set(endog_names).difference(set(factors.keys()))
    if len(missing):
        raise ValueError(f'If a `factors` dictionary is provided, then it must include entries for each observed variable. Missing variables are: {missing}.')
    factor_names = {}
    for key, value in factors.items():
        if isinstance(value, str):
            factor_names[value] = 0
        else:
            factor_names.update({v: 0 for v in value})
    factor_names = list(factor_names.keys())
    k_factors = len(factor_names)
    endog_factor_map = pd.DataFrame(np.zeros((self.k_endog, k_factors), dtype=bool), index=pd.Index(endog_names, name='endog'), columns=pd.Index(factor_names, name='factor'))
    for key, value in factors.items():
        endog_factor_map.loc[key, value] = True
    return endog_factor_map