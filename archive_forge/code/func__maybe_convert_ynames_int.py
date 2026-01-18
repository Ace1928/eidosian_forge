from statsmodels.compat.pandas import Appender
import warnings
import numpy as np
from pandas import MultiIndex, get_dummies
from scipy import special, stats
from scipy.special import digamma, gammaln, loggamma, polygamma
from scipy.stats import nbinom
from statsmodels.base.data import handle_data  # for mnlogit
from statsmodels.base.l1_slsqp import fit_l1_slsqp
import statsmodels.base.model as base
import statsmodels.base.wrapper as wrap
from statsmodels.base._constraints import fit_constrained_wrap
import statsmodels.base._parameter_inference as pinfer
from statsmodels.base import _prediction_inference as pred
from statsmodels.distributions import genpoisson_p
import statsmodels.regression.linear_model as lm
from statsmodels.tools import data as data_tools, tools
from statsmodels.tools.decorators import cache_readonly
from statsmodels.tools.numdiff import approx_fprime_cs
from statsmodels.tools.sm_exceptions import (
@staticmethod
def _maybe_convert_ynames_int(ynames):
    issue_warning = False
    msg = 'endog contains values are that not int-like. Uses string representation of value. Use integer-valued endog to suppress this warning.'
    for i in ynames:
        try:
            if ynames[i] % 1 == 0:
                ynames[i] = str(int(ynames[i]))
            else:
                issue_warning = True
                ynames[i] = str(ynames[i])
        except TypeError:
            ynames[i] = str(ynames[i])
    if issue_warning:
        warnings.warn(msg, SpecificationWarning)
    return ynames