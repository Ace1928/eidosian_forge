import warnings
import numpy as np
from scipy.optimize import OptimizeResult, minimize
from scipy.optimize._optimize import _status_message, _wrap_callback
from scipy._lib._util import check_random_state, MapWrapper, _FunctionWrapper
from scipy.optimize._constraints import (Bounds, new_bounds_to_old,
from scipy.sparse import issparse
def _calculate_population_energies(self, population):
    """
        Calculate the energies of a population.

        Parameters
        ----------
        population : ndarray
            An array of parameter vectors normalised to [0, 1] using lower
            and upper limits. Has shape ``(np.size(population, 0), N)``.

        Returns
        -------
        energies : ndarray
            An array of energies corresponding to each population member. If
            maxfun will be exceeded during this call, then the number of
            function evaluations will be reduced and energies will be
            right-padded with np.inf. Has shape ``(np.size(population, 0),)``
        """
    num_members = np.size(population, 0)
    S = min(num_members, self.maxfun - self._nfev)
    energies = np.full(num_members, np.inf)
    parameters_pop = self._scale_parameters(population)
    try:
        calc_energies = list(self._mapwrapper(self.func, parameters_pop[0:S]))
        calc_energies = np.squeeze(calc_energies)
    except (TypeError, ValueError) as e:
        raise RuntimeError("The map-like callable must be of the form f(func, iterable), returning a sequence of numbers the same length as 'iterable'") from e
    if calc_energies.size != S:
        if self.vectorized:
            raise RuntimeError('The vectorized function must return an array of shape (S,) when given an array of shape (len(x), S)')
        raise RuntimeError('func(x, *args) must return a scalar value')
    energies[0:S] = calc_energies
    if self.vectorized:
        self._nfev += 1
    else:
        self._nfev += S
    return energies