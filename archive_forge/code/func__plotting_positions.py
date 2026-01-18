import warnings
from collections import namedtuple
import numpy as np
from scipy import optimize, stats
from scipy._lib._util import check_random_state
def _plotting_positions(self, n, a=0.5):
    k = np.arange(1, n + 1)
    return (k - a) / (n + 1 - 2 * a)