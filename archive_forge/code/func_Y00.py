import numpy as np
from numpy.testing import assert_allclose
import scipy.special as sc
def Y00(theta, phi):
    return 0.5 * np.sqrt(1 / np.pi)