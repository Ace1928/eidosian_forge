import numpy as np
from ._hessian_update_strategy import BFGS
from ._differentiable_functions import (
from ._optimize import OptimizeWarning
from warnings import warn, catch_warnings, simplefilter, filterwarnings
from scipy.sparse import issparse
def _input_validation(self):
    try:
        res = np.broadcast_arrays(self.lb, self.ub, self.keep_feasible)
        self.lb, self.ub, self.keep_feasible = res
    except ValueError:
        message = '`lb`, `ub`, and `keep_feasible` must be broadcastable.'
        raise ValueError(message)