import numpy as np
import math
import inspect
import scipy.optimize
from scipy._lib._util import check_random_state
def accept_reject(self, res_new, res_old):
    """
        Assuming the local search underlying res_new was successful:
        If new energy is lower than old, it will always be accepted.
        If new is higher than old, there is a chance it will be accepted,
        less likely for larger differences.
        """
    with np.errstate(invalid='ignore'):
        prod = -(res_new.fun - res_old.fun) * self.beta
        w = math.exp(min(0, prod))
    rand = self.random_gen.uniform()
    return w >= rand and (res_new.success or not res_old.success)