import numpy as np
from scipy.optimize import OptimizeResult
from scipy.optimize import minimize, Bounds
from scipy.special import gammaln
from scipy._lib._util import check_random_state
from scipy.optimize._constraints import new_bounds_to_old
class EnergyState:
    """
    Class used to record the energy state. At any time, it knows what is the
    currently used coordinates and the most recent best location.

    Parameters
    ----------
    lower : array_like
        A 1-D NumPy ndarray containing lower bounds for generating an initial
        random components in the `reset` method.
    upper : array_like
        A 1-D NumPy ndarray containing upper bounds for generating an initial
        random components in the `reset` method
        components. Neither NaN or inf are allowed.
    callback : callable, ``callback(x, f, context)``, optional
        A callback function which will be called for all minima found.
        ``x`` and ``f`` are the coordinates and function value of the
        latest minimum found, and `context` has value in [0, 1, 2]
    """
    MAX_REINIT_COUNT = 1000

    def __init__(self, lower, upper, callback=None):
        self.ebest = None
        self.current_energy = None
        self.current_location = None
        self.xbest = None
        self.lower = lower
        self.upper = upper
        self.callback = callback

    def reset(self, func_wrapper, rand_gen, x0=None):
        """
        Initialize current location is the search domain. If `x0` is not
        provided, a random location within the bounds is generated.
        """
        if x0 is None:
            self.current_location = rand_gen.uniform(self.lower, self.upper, size=len(self.lower))
        else:
            self.current_location = np.copy(x0)
        init_error = True
        reinit_counter = 0
        while init_error:
            self.current_energy = func_wrapper.fun(self.current_location)
            if self.current_energy is None:
                raise ValueError('Objective function is returning None')
            if not np.isfinite(self.current_energy) or np.isnan(self.current_energy):
                if reinit_counter >= EnergyState.MAX_REINIT_COUNT:
                    init_error = False
                    message = 'Stopping algorithm because function create NaN or (+/-) infinity values even with trying new random parameters'
                    raise ValueError(message)
                self.current_location = rand_gen.uniform(self.lower, self.upper, size=self.lower.size)
                reinit_counter += 1
            else:
                init_error = False
            if self.ebest is None and self.xbest is None:
                self.ebest = self.current_energy
                self.xbest = np.copy(self.current_location)

    def update_best(self, e, x, context):
        self.ebest = e
        self.xbest = np.copy(x)
        if self.callback is not None:
            val = self.callback(x, e, context)
            if val is not None:
                if val:
                    return 'Callback function requested to stop early by returning True'

    def update_current(self, e, x):
        self.current_energy = e
        self.current_location = np.copy(x)