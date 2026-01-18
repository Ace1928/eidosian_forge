import numpy as np
from scipy.optimize import OptimizeResult
from scipy.optimize import minimize, Bounds
from scipy.special import gammaln
from scipy._lib._util import check_random_state
from scipy.optimize._constraints import new_bounds_to_old
class StrategyChain:
    """
    Class that implements within a Markov chain the strategy for location
    acceptance and local search decision making.

    Parameters
    ----------
    acceptance_param : float
        Parameter for acceptance distribution. It is used to control the
        probability of acceptance. The lower the acceptance parameter, the
        smaller the probability of acceptance. Default value is -5.0 with
        a range (-1e4, -5].
    visit_dist : VisitingDistribution
        Instance of `VisitingDistribution` class.
    func_wrapper : ObjectiveFunWrapper
        Instance of `ObjectiveFunWrapper` class.
    minimizer_wrapper: LocalSearchWrapper
        Instance of `LocalSearchWrapper` class.
    rand_gen : {None, int, `numpy.random.Generator`,
                `numpy.random.RandomState`}, optional

        If `seed` is None (or `np.random`), the `numpy.random.RandomState`
        singleton is used.
        If `seed` is an int, a new ``RandomState`` instance is used,
        seeded with `seed`.
        If `seed` is already a ``Generator`` or ``RandomState`` instance then
        that instance is used.
    energy_state: EnergyState
        Instance of `EnergyState` class.

    """

    def __init__(self, acceptance_param, visit_dist, func_wrapper, minimizer_wrapper, rand_gen, energy_state):
        self.emin = energy_state.current_energy
        self.xmin = np.array(energy_state.current_location)
        self.energy_state = energy_state
        self.acceptance_param = acceptance_param
        self.visit_dist = visit_dist
        self.func_wrapper = func_wrapper
        self.minimizer_wrapper = minimizer_wrapper
        self.not_improved_idx = 0
        self.not_improved_max_idx = 1000
        self._rand_gen = rand_gen
        self.temperature_step = 0
        self.K = 100 * len(energy_state.current_location)

    def accept_reject(self, j, e, x_visit):
        r = self._rand_gen.uniform()
        pqv_temp = 1.0 - (1.0 - self.acceptance_param) * (e - self.energy_state.current_energy) / self.temperature_step
        if pqv_temp <= 0.0:
            pqv = 0.0
        else:
            pqv = np.exp(np.log(pqv_temp) / (1.0 - self.acceptance_param))
        if r <= pqv:
            self.energy_state.update_current(e, x_visit)
            self.xmin = np.copy(self.energy_state.current_location)
        if self.not_improved_idx >= self.not_improved_max_idx:
            if j == 0 or self.energy_state.current_energy < self.emin:
                self.emin = self.energy_state.current_energy
                self.xmin = np.copy(self.energy_state.current_location)

    def run(self, step, temperature):
        self.temperature_step = temperature / float(step + 1)
        self.not_improved_idx += 1
        for j in range(self.energy_state.current_location.size * 2):
            if j == 0:
                if step == 0:
                    self.energy_state_improved = True
                else:
                    self.energy_state_improved = False
            x_visit = self.visit_dist.visiting(self.energy_state.current_location, j, temperature)
            e = self.func_wrapper.fun(x_visit)
            if e < self.energy_state.current_energy:
                self.energy_state.update_current(e, x_visit)
                if e < self.energy_state.ebest:
                    val = self.energy_state.update_best(e, x_visit, 0)
                    if val is not None:
                        if val:
                            return val
                    self.energy_state_improved = True
                    self.not_improved_idx = 0
            else:
                self.accept_reject(j, e, x_visit)
            if self.func_wrapper.nfev >= self.func_wrapper.maxfun:
                return 'Maximum number of function call reached during annealing'

    def local_search(self):
        if self.energy_state_improved:
            e, x = self.minimizer_wrapper.local_search(self.energy_state.xbest, self.energy_state.ebest)
            if e < self.energy_state.ebest:
                self.not_improved_idx = 0
                val = self.energy_state.update_best(e, x, 1)
                if val is not None:
                    if val:
                        return val
                self.energy_state.update_current(e, x)
            if self.func_wrapper.nfev >= self.func_wrapper.maxfun:
                return 'Maximum number of function call reached during local search'
        do_ls = False
        if self.K < 90 * len(self.energy_state.current_location):
            pls = np.exp(self.K * (self.energy_state.ebest - self.energy_state.current_energy) / self.temperature_step)
            if pls >= self._rand_gen.uniform():
                do_ls = True
        if self.not_improved_idx >= self.not_improved_max_idx:
            do_ls = True
        if do_ls:
            e, x = self.minimizer_wrapper.local_search(self.xmin, self.emin)
            self.xmin = np.copy(x)
            self.emin = e
            self.not_improved_idx = 0
            self.not_improved_max_idx = self.energy_state.current_location.size
            if e < self.energy_state.ebest:
                val = self.energy_state.update_best(self.emin, self.xmin, 2)
                if val is not None:
                    if val:
                        return val
                self.energy_state.update_current(e, x)
            if self.func_wrapper.nfev >= self.func_wrapper.maxfun:
                return 'Maximum number of function call reached during dual annealing'