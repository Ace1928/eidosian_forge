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
def _print_verbose_msg_init_beg(self, n_init):
    """Print verbose message on initialization."""
    if self.verbose == 1:
        print('Initialization %d' % n_init)
    elif self.verbose >= 2:
        print('Initialization %d' % n_init)
        self._init_prev_time = time()
        self._iter_prev_time = self._init_prev_time