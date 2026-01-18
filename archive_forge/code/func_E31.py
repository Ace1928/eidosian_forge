import numpy as np
from numpy.testing import (assert_equal, assert_almost_equal, assert_allclose,
from scipy.special._testutils import assert_func_equal
from scipy.special import ellip_harm, ellip_harm_2, ellip_normal
from scipy.integrate import IntegrationWarning
from numpy import sqrt, pi
def E31(h2, k2, s):
    return s * s * s - s / 5 * (2 * (h2 + k2) + sqrt(4 * (h2 + k2) * (h2 + k2) - 15 * h2 * k2))