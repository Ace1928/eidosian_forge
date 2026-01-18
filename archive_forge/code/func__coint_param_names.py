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
def _coint_param_names(self):
    """
        Returns parameter names (for beta and deterministics) for the summary.

        Returns
        -------
        param_names : list of str
            Returns a list of parameter names for the cointegration matrix
            as well as deterministic terms inside the cointegration relation
            (if present in the model).
        """
    param_names = []
    param_names += [('beta.%d.' + self.load_coef_repr + '%d') % (j + 1, i + 1) for i in range(self.coint_rank) for j in range(self.neqs)]
    if 'ci' in self.deterministic:
        param_names += ['const.' + self.load_coef_repr + '%d' % (i + 1) for i in range(self.coint_rank)]
    if 'li' in self.deterministic:
        param_names += ['lin_trend.' + self.load_coef_repr + '%d' % (i + 1) for i in range(self.coint_rank)]
    if self.exog_coint is not None:
        param_names += ['exog_coint%d.%s' % (n + 1, exog_no) for exog_no in range(1, self.exog_coint.shape[1] + 1) for n in range(self.neqs)]
    return param_names