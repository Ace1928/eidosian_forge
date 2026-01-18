from collections import defaultdict
import numpy as np
from numpy import hstack, vstack
from numpy.linalg import inv, svd
import scipy
import scipy.stats
from statsmodels.iolib.summary import Summary
from statsmodels.iolib.table import SimpleTable
from statsmodels.tools.decorators import cache_readonly
from statsmodels.tools.sm_exceptions import HypothesisTestWarning
from statsmodels.tools.validation import string_like
import statsmodels.tsa.base.tsa_model as tsbase
from statsmodels.tsa.coint_tables import c_sja, c_sjt
from statsmodels.tsa.tsatools import duplication_matrix, lagmat, vec
from statsmodels.tsa.vector_ar.hypothesis_test_results import (
import statsmodels.tsa.vector_ar.irf as irf
import statsmodels.tsa.vector_ar.plotting as plot
from statsmodels.tsa.vector_ar.util import get_index, seasonal_dummies
from statsmodels.tsa.vector_ar.var_model import (
@property
def _lagged_param_names(self):
    """
        Returns parameter names (for Gamma and deterministics) for the summary.

        Returns
        -------
        param_names : list of str
            Returns a list of parameter names for the lagged endogenous
            parameters which are called :math:`\\Gamma` in [1]_
            (see chapter 6).
            If present in the model, also names for deterministic terms outside
            the cointegration relation are returned. They name the elements of
            the matrix C in [1]_ (p. 299).

        References
        ----------
        .. [1] Lütkepohl, H. 2005. *New Introduction to Multiple Time Series Analysis*. Springer.
        """
    param_names = []
    if 'co' in self.deterministic:
        param_names += ['const.%s' % n for n in self.endog_names]
    if self.seasons > 0:
        param_names += ['season%d.%s' % (s, n) for s in range(1, self.seasons) for n in self.endog_names]
    if 'lo' in self.deterministic:
        param_names += ['lin_trend.%s' % n for n in self.endog_names]
    if self.exog is not None:
        param_names += ['exog%d.%s' % (exog_no, n) for exog_no in range(1, self.exog.shape[1] + 1) for n in self.endog_names]
    param_names += ['L%d.%s.%s' % (i + 1, n1, n2) for n2 in self.endog_names for i in range(self.k_ar_diff) for n1 in self.endog_names]
    return param_names