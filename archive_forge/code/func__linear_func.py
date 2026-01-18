import itertools
import numpy as np
from numpy.testing import assert_allclose
from scipy.integrate import ode
def _linear_func(t, y, a):
    """Linear system dy/dt = a * y"""
    return a.dot(y)