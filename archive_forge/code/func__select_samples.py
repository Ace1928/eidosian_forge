import warnings
import numpy as np
from scipy.optimize import OptimizeResult, minimize
from scipy.optimize._optimize import _status_message, _wrap_callback
from scipy._lib._util import check_random_state, MapWrapper, _FunctionWrapper
from scipy.optimize._constraints import (Bounds, new_bounds_to_old,
from scipy.sparse import issparse
def _select_samples(self, candidate, number_samples):
    """
        obtain random integers from range(self.num_population_members),
        without replacement. You can't have the original candidate either.
        """
    pool = np.arange(self.num_population_members)
    self.random_number_generator.shuffle(pool)
    idxs = []
    while len(idxs) < number_samples and len(pool) > 0:
        idx = pool[0]
        pool = pool[1:]
        if idx != candidate:
            idxs.append(idx)
    return idxs