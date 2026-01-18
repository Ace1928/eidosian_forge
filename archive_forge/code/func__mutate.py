import warnings
import numpy as np
from scipy.optimize import OptimizeResult, minimize
from scipy.optimize._optimize import _status_message, _wrap_callback
from scipy._lib._util import check_random_state, MapWrapper, _FunctionWrapper
from scipy.optimize._constraints import (Bounds, new_bounds_to_old,
from scipy.sparse import issparse
def _mutate(self, candidate):
    """Create a trial vector based on a mutation strategy."""
    rng = self.random_number_generator
    if callable(self.strategy):
        _population = self._scale_parameters(self.population)
        trial = np.array(self.strategy(candidate, _population, rng=rng), dtype=float)
        if trial.shape != (self.parameter_count,):
            raise RuntimeError('strategy must have signature f(candidate: int, population: np.ndarray, rng=None) returning an array of shape (N,)')
        return self._unscale_parameters(trial)
    trial = np.copy(self.population[candidate])
    fill_point = rng.choice(self.parameter_count)
    if self.strategy in ['currenttobest1exp', 'currenttobest1bin']:
        bprime = self.mutation_func(candidate, self._select_samples(candidate, 5))
    else:
        bprime = self.mutation_func(self._select_samples(candidate, 5))
    if self.strategy in self._binomial:
        crossovers = rng.uniform(size=self.parameter_count)
        crossovers = crossovers < self.cross_over_probability
        crossovers[fill_point] = True
        trial = np.where(crossovers, bprime, trial)
        return trial
    elif self.strategy in self._exponential:
        i = 0
        crossovers = rng.uniform(size=self.parameter_count)
        crossovers = crossovers < self.cross_over_probability
        crossovers[0] = True
        while i < self.parameter_count and crossovers[i]:
            trial[fill_point] = bprime[fill_point]
            fill_point = (fill_point + 1) % self.parameter_count
            i += 1
        return trial