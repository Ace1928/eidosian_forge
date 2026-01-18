from statsmodels.compat.python import lrange
import scipy
import numpy as np
from itertools import product
from statsmodels.genmod.families import Gaussian
from statsmodels.genmod.generalized_estimating_equations import GEE
from statsmodels.genmod.cov_struct import Autoregressive, Nested
class GEE_simulator:
    ngroups = None
    error_sd = None
    params = None
    dparams = None
    exog = None
    endog = None
    time = None
    group = None
    group_size_range = [4, 11]

    def print_dparams(self, dparams_est):
        raise NotImplementedError