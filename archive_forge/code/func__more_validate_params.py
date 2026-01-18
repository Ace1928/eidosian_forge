import warnings
from abc import ABCMeta, abstractmethod
from numbers import Integral, Real
import numpy as np
from ..base import (
from ..exceptions import ConvergenceWarning
from ..model_selection import ShuffleSplit, StratifiedShuffleSplit
from ..utils import check_random_state, compute_class_weight, deprecated
from ..utils._param_validation import Hidden, Interval, StrOptions
from ..utils.extmath import safe_sparse_dot
from ..utils.metaestimators import available_if
from ..utils.multiclass import _check_partial_fit_first_call
from ..utils.parallel import Parallel, delayed
from ..utils.validation import _check_sample_weight, check_is_fitted
from ._base import LinearClassifierMixin, SparseCoefMixin, make_dataset
from ._sgd_fast import (
def _more_validate_params(self, for_partial_fit=False):
    """Validate input params."""
    if self.early_stopping and for_partial_fit:
        raise ValueError('early_stopping should be False with partial_fit')
    if self.learning_rate in ('constant', 'invscaling', 'adaptive') and self.eta0 <= 0.0:
        raise ValueError('eta0 must be > 0')
    if self.learning_rate == 'optimal' and self.alpha == 0:
        raise ValueError("alpha must be > 0 since learning_rate is 'optimal'. alpha is used to compute the optimal learning rate.")
    self._get_penalty_type(self.penalty)
    self._get_learning_rate_type(self.learning_rate)