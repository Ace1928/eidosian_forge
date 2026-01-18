import warnings
import numpy as np
from scipy.optimize import OptimizeResult, minimize
from scipy.optimize._optimize import _status_message, _wrap_callback
from scipy._lib._util import check_random_state, MapWrapper, _FunctionWrapper
from scipy.optimize._constraints import (Bounds, new_bounds_to_old,
from scipy.sparse import issparse
def _calculate_population_feasibilities(self, population):
    """
        Calculate the feasibilities of a population.

        Parameters
        ----------
        population : ndarray
            An array of parameter vectors normalised to [0, 1] using lower
            and upper limits. Has shape ``(np.size(population, 0), N)``.

        Returns
        -------
        feasible, constraint_violation : ndarray, ndarray
            Boolean array of feasibility for each population member, and an
            array of the constraint violation for each population member.
            constraint_violation has shape ``(np.size(population, 0), M)``,
            where M is the number of constraints.
        """
    num_members = np.size(population, 0)
    if not self._wrapped_constraints:
        return (np.ones(num_members, bool), np.zeros((num_members, 1)))
    parameters_pop = self._scale_parameters(population)
    if self.vectorized:
        constraint_violation = np.array(self._constraint_violation_fn(parameters_pop))
    else:
        constraint_violation = np.array([self._constraint_violation_fn(x) for x in parameters_pop])
        constraint_violation = constraint_violation[:, 0]
    feasible = ~(np.sum(constraint_violation, axis=1) > 0)
    return (feasible, constraint_violation)