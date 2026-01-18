import warnings
import numpy as np
from scipy.optimize import OptimizeResult, minimize
from scipy.optimize._optimize import _status_message, _wrap_callback
from scipy._lib._util import check_random_state, MapWrapper, _FunctionWrapper
from scipy.optimize._constraints import (Bounds, new_bounds_to_old,
from scipy.sparse import issparse
def _promote_lowest_energy(self):
    idx = np.arange(self.num_population_members)
    feasible_solutions = idx[self.feasible]
    if feasible_solutions.size:
        idx_t = np.argmin(self.population_energies[feasible_solutions])
        l = feasible_solutions[idx_t]
    else:
        l = np.argmin(np.sum(self.constraint_violation, axis=1))
    self.population_energies[[0, l]] = self.population_energies[[l, 0]]
    self.population[[0, l], :] = self.population[[l, 0], :]
    self.feasible[[0, l]] = self.feasible[[l, 0]]
    self.constraint_violation[[0, l], :] = self.constraint_violation[[l, 0], :]