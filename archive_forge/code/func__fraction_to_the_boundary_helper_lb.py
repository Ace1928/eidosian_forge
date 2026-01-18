from pyomo.contrib.pynumero.interfaces.utils import (
import numpy as np
import logging
import time
from pyomo.contrib.pynumero.linalg.base import LinearSolverStatus
from pyomo.common.timing import HierarchicalTimer
import enum
def _fraction_to_the_boundary_helper_lb(tau, x, delta_x, xl):
    delta_x_mod = delta_x.copy()
    delta_x_mod[delta_x_mod == 0] = 1
    alpha = -tau * (x - xl) / delta_x_mod
    alpha[delta_x >= 0] = np.inf
    if alpha.size == 0:
        return 1
    else:
        return min(alpha.min(), 1)