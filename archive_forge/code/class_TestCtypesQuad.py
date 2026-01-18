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
class TestCtypesQuad:

    def setup_method(self):
        if sys.platform == 'win32':
            files = ['api-ms-win-crt-math-l1-1-0.dll']
        elif sys.platform == 'darwin':
            files = ['libm.dylib']
        else:
            files = ['libm.so', 'libm.so.6']
        for file in files:
            try:
                self.lib = ctypes.CDLL(file)
                break
            except OSError:
                pass
        else:
            pytest.skip("Ctypes can't import libm.so")
        restype = ctypes.c_double
        argtypes = (ctypes.c_double,)
        for name in ['sin', 'cos', 'tan']:
            func = getattr(self.lib, name)
            func.restype = restype
            func.argtypes = argtypes

    def test_typical(self):
        assert_quad(quad(self.lib.sin, 0, 5), quad(math.sin, 0, 5)[0])
        assert_quad(quad(self.lib.cos, 0, 5), quad(math.cos, 0, 5)[0])
        assert_quad(quad(self.lib.tan, 0, 1), quad(math.tan, 0, 1)[0])

    def test_ctypes_sine(self):
        quad(LowLevelCallable(sine_ctypes), 0, 1)

    def test_ctypes_variants(self):
        sin_0 = get_clib_test_routine('_sin_0', ctypes.c_double, ctypes.c_double, ctypes.c_void_p)
        sin_1 = get_clib_test_routine('_sin_1', ctypes.c_double, ctypes.c_int, ctypes.POINTER(ctypes.c_double), ctypes.c_void_p)
        sin_2 = get_clib_test_routine('_sin_2', ctypes.c_double, ctypes.c_double)
        sin_3 = get_clib_test_routine('_sin_3', ctypes.c_double, ctypes.c_int, ctypes.POINTER(ctypes.c_double))
        sin_4 = get_clib_test_routine('_sin_3', ctypes.c_double, ctypes.c_int, ctypes.c_double)
        all_sigs = [sin_0, sin_1, sin_2, sin_3, sin_4]
        legacy_sigs = [sin_2, sin_4]
        legacy_only_sigs = [sin_4]
        for j, func in enumerate(all_sigs):
            callback = LowLevelCallable(func)
            if func in legacy_only_sigs:
                pytest.raises(ValueError, quad, callback, 0, pi)
            else:
                assert_allclose(quad(callback, 0, pi)[0], 2.0)
        for j, func in enumerate(legacy_sigs):
            if func in legacy_sigs:
                assert_allclose(quad(func, 0, pi)[0], 2.0)
            else:
                pytest.raises(ValueError, quad, func, 0, pi)