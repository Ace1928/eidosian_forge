import warnings
import numpy as np
from scipy.optimize import OptimizeResult, minimize
from scipy.optimize._optimize import _status_message, _wrap_callback
from scipy._lib._util import check_random_state, MapWrapper, _FunctionWrapper
from scipy.optimize._constraints import (Bounds, new_bounds_to_old,
from scipy.sparse import issparse
def _currenttobest1(self, candidate, samples):
    """currenttobest1bin, currenttobest1exp"""
    r0, r1 = samples[:2]
    bprime = self.population[candidate] + self.scale * (self.population[0] - self.population[candidate] + self.population[r0] - self.population[r1])
    return bprime