import sys
import math
import numpy as np
from numpy import sqrt, cos, sin, arctan, exp, log, pi
from numpy.testing import (assert_,
import pytest
from scipy.integrate import quad, dblquad, tplquad, nquad
from scipy.special import erf, erfc
from scipy._lib._ccallback import LowLevelCallable
import ctypes
import ctypes.util
from scipy._lib._ccallback_c import sine_ctypes
import scipy.integrate._test_multivariate as clib_test
class TestMultivariateCtypesQuad:

    def setup_method(self):
        restype = ctypes.c_double
        argtypes = (ctypes.c_int, ctypes.c_double)
        for name in ['_multivariate_typical', '_multivariate_indefinite', '_multivariate_sin']:
            func = get_clib_test_routine(name, restype, *argtypes)
            setattr(self, name, func)

    def test_typical(self):
        assert_quad(quad(self._multivariate_typical, 0, pi, (2, 1.8)), 0.30614353532540295)

    def test_indefinite(self):
        assert_quad(quad(self._multivariate_indefinite, 0, np.inf), 0.5772156649015329)

    def test_threadsafety(self):

        def threadsafety(y):
            return y + quad(self._multivariate_sin, 0, 1)[0]
        assert_quad(quad(threadsafety, 0, 1), 0.9596976941318602)