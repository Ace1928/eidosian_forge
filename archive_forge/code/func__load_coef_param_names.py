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
def _load_coef_param_names(self):
    """
        Returns parameter names (for alpha) for the summary.

        Returns
        -------
        param_names : list of str
            Returns a list of parameter names for the loading coefficients
            which are called :math:`\\alpha` in [1]_ (see chapter 6).

        References
        ----------
        .. [1] Lütkepohl, H. 2005. *New Introduction to Multiple Time Series Analysis*. Springer.
        """
    param_names = []
    if self.coint_rank == 0:
        return None
    param_names += [self.load_coef_repr + '%d.%s' % (i + 1, self.endog_names[j]) for j in range(self.neqs) for i in range(self.coint_rank)]
    return param_names