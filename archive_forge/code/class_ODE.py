import numpy as np
from numpy import (arange, zeros, array, dot, sqrt, cos, sin, eye, pi, exp,
from numpy.testing import (
from pytest import raises as assert_raises
from scipy.integrate import odeint, ode, complex_ode
class ODE:
    """
    ODE problem
    """
    stiff = False
    cmplx = False
    stop_t = 1
    z0 = []
    lband = None
    uband = None
    atol = 1e-06
    rtol = 1e-05