import numpy as np
from numpy.testing import assert_allclose
import scipy.special as sc
def Yn11(theta, phi):
    return 0.5 * np.sqrt(3 / (2 * np.pi)) * np.exp(-1j * theta) * np.sin(phi)