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
def _get_endog_names(self, truncate=None, as_string=None):
    if truncate is None:
        truncate = False if as_string is False or self.k_endog == 1 else 24
    if as_string is False and truncate is not False:
        raise ValueError('Can only truncate endog names if they are returned as a string.')
    if as_string is None:
        as_string = truncate is not False
    endog_names = self.endog_names
    if not isinstance(endog_names, list):
        endog_names = [endog_names]
    if as_string:
        endog_names = [str(name) for name in endog_names]
    if truncate is not False:
        n = truncate
        endog_names = [name if len(name) <= n else name[:n] + '...' for name in endog_names]
    return endog_names