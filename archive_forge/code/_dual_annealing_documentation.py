import numpy as np
from scipy.optimize import OptimizeResult
from scipy.optimize import minimize, Bounds
from scipy.special import gammaln
from scipy._lib._util import check_random_state
from scipy.optimize._constraints import new_bounds_to_old

        Initialize current location is the search domain. If `x0` is not
        provided, a random location within the bounds is generated.
        