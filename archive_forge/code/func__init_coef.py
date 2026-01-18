import warnings
from abc import ABCMeta, abstractmethod
from itertools import chain
from numbers import Integral, Real
import numpy as np
import scipy.optimize
from ..base import (
from ..exceptions import ConvergenceWarning
from ..metrics import accuracy_score, r2_score
from ..model_selection import train_test_split
from ..preprocessing import LabelBinarizer
from ..utils import (
from ..utils._param_validation import Interval, Options, StrOptions
from ..utils.extmath import safe_sparse_dot
from ..utils.metaestimators import available_if
from ..utils.multiclass import (
from ..utils.optimize import _check_optimize_result
from ..utils.validation import check_is_fitted
from ._base import ACTIVATIONS, DERIVATIVES, LOSS_FUNCTIONS
from ._stochastic_optimizers import AdamOptimizer, SGDOptimizer
def _init_coef(self, fan_in, fan_out, dtype):
    factor = 6.0
    if self.activation == 'logistic':
        factor = 2.0
    init_bound = np.sqrt(factor / (fan_in + fan_out))
    coef_init = self._random_state.uniform(-init_bound, init_bound, (fan_in, fan_out))
    intercept_init = self._random_state.uniform(-init_bound, init_bound, fan_out)
    coef_init = coef_init.astype(dtype, copy=False)
    intercept_init = intercept_init.astype(dtype, copy=False)
    return (coef_init, intercept_init)