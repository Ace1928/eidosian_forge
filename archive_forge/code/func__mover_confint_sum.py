import numpy as np
from numpy.testing import assert_allclose
def _mover_confint_sum(stat, ci):
    stat_ = stat.sum(0)
    low_half = np.sqrt(np.sum((stat_ - ci[0]) ** 2))
    upp_half = np.sqrt(np.sum((stat_ - ci[1]) ** 2))
    ci = (stat - low_half, stat + upp_half)
    return ci