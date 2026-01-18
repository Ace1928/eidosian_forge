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
def _print_verbose_msg_iter_end(self, n_iter, diff_ll):
    """Print verbose message on initialization."""
    if n_iter % self.verbose_interval == 0:
        if self.verbose == 1:
            print('  Iteration %d' % n_iter)
        elif self.verbose >= 2:
            cur_time = time()
            print('  Iteration %d\t time lapse %.5fs\t ll change %.5f' % (n_iter, cur_time - self._iter_prev_time, diff_ll))
            self._iter_prev_time = cur_time