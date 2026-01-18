import warnings
from abc import ABCMeta, abstractmethod
from numbers import Integral, Real
from time import time
import numpy as np
from scipy.special import logsumexp
from .. import cluster
from ..base import BaseEstimator, DensityMixin, _fit_context
from ..cluster import kmeans_plusplus
from ..exceptions import ConvergenceWarning
from ..utils import check_random_state
from ..utils._param_validation import Interval, StrOptions
from ..utils.validation import check_is_fitted
def _print_verbose_msg_init_end(self, ll):
    """Print verbose message on the end of iteration."""
    if self.verbose == 1:
        print('Initialization converged: %s' % self.converged_)
    elif self.verbose >= 2:
        print('Initialization converged: %s\t time lapse %.5fs\t ll %.5f' % (self.converged_, time() - self._init_prev_time, ll))