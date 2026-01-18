import itertools
import numpy as np
from numpy.testing import assert_allclose
from scipy.integrate import ode
def _linear_banded_jac(t, y, a):
    """Banded Jacobian."""
    ml, mu = _band_count(a)
    bjac = [np.r_[[0] * k, np.diag(a, k)] for k in range(mu, 0, -1)]
    bjac.append(np.diag(a))
    for k in range(-1, -ml - 1, -1):
        bjac.append(np.r_[np.diag(a, k), [0] * -k])
    return bjac