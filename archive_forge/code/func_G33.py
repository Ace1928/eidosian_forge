import numpy as np
from numpy.testing import (assert_equal, assert_almost_equal, assert_allclose,
from scipy.special._testutils import assert_func_equal
from scipy.special import ellip_harm, ellip_harm_2, ellip_normal
from scipy.integrate import IntegrationWarning
from numpy import sqrt, pi
def G33(h2, k2):
    res = 6 * h2 ** 4 + 16 * k2 ** 4 - 12 * h2 ** 3 * k2 - 28 * h2 * k2 ** 3 + 34 * h2 ** 2 * k2 ** 2 + sqrt(h2 ** 2 + 4 * k2 ** 2 - h2 * k2) * (6 * h2 ** 3 + 8 * k2 ** 3 - 9 * h2 ** 2 * k2 - 13 * h2 * k2 ** 2)
    return 16 * pi / 13125 * h2 * (k2 - h2) * res