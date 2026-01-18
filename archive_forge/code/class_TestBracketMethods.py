import pytest
from functools import lru_cache
from numpy.testing import (assert_warns, assert_,
import numpy as np
from numpy import finfo, power, nan, isclose, sqrt, exp, sin, cos
from scipy import stats, optimize
from scipy.optimize import (_zeros_py as zeros, newton, root_scalar,
from scipy._lib._util import getfullargspec_no_self as _getfullargspec
from scipy.optimize._tstutils import get_tests, functions as tstutils_functions
class TestBracketMethods(TestScalarRootFinders):

    @pytest.mark.parametrize('method', bracket_methods)
    @pytest.mark.parametrize('function', tstutils_functions)
    def test_basic_root_scalar(self, method, function):
        a, b = (0.5, sqrt(3))
        r = root_scalar(function, method=method.__name__, bracket=[a, b], x0=a, xtol=self.xtol, rtol=self.rtol)
        assert r.converged
        assert_allclose(r.root, 1.0, atol=self.xtol, rtol=self.rtol)
        assert r.method == method.__name__

    @pytest.mark.parametrize('method', bracket_methods)
    @pytest.mark.parametrize('function', tstutils_functions)
    def test_basic_individual(self, method, function):
        a, b = (0.5, sqrt(3))
        root, r = method(function, a, b, xtol=self.xtol, rtol=self.rtol, full_output=True)
        assert r.converged
        assert_allclose(root, 1.0, atol=self.xtol, rtol=self.rtol)

    @pytest.mark.parametrize('method', bracket_methods)
    def test_aps_collection(self, method):
        self.run_collection('aps', method, method.__name__, smoothness=1)

    @pytest.mark.parametrize('method', [zeros.bisect, zeros.ridder, zeros.toms748])
    def test_chandrupatla_collection(self, method):
        known_fail = {'fun7.4'} if method == zeros.ridder else {}
        self.run_collection('chandrupatla', method, method.__name__, known_fail=known_fail)

    @pytest.mark.parametrize('method', bracket_methods)
    def test_lru_cached_individual(self, method):
        a, b = (-1, 1)
        root, r = method(f_lrucached, a, b, full_output=True)
        assert r.converged
        assert_allclose(root, 0)