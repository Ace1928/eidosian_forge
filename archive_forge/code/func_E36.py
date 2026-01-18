import numpy as np
from numpy.testing import (assert_equal, assert_almost_equal, assert_allclose,
from scipy.special._testutils import assert_func_equal
from scipy.special import ellip_harm, ellip_harm_2, ellip_normal
from scipy.integrate import IntegrationWarning
from numpy import sqrt, pi
def E36(h2, k2, s):
    return sqrt(abs(s * s - k2)) * (s * s - 1 / 5 * (2 * h2 + k2 - sqrt(abs((2 * h2 + k2) * (2 * h2 + k2) - 5 * h2 * k2))))