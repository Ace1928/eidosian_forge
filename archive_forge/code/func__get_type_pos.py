import copy
import numpy as np
from scipy import optimize
from scipy.stats.mstats import mquantiles
from . import kernels
def _get_type_pos(var_type):
    ix_cont = np.array([c == 'c' for c in var_type])
    ix_ord = np.array([c == 'o' for c in var_type])
    ix_unord = np.array([c == 'u' for c in var_type])
    return (ix_cont, ix_ord, ix_unord)