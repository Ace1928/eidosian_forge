import logging
import sys
import numpy
import numpy as np
import time
from multiprocessing import Pool
from numpy.testing import assert_allclose, IS_PYPY
import pytest
from pytest import raises as assert_raises, warns
from scipy.optimize import (shgo, Bounds, minimize_scalar, minimize, rosen,
from scipy.optimize._constraints import new_constraint_to_old
from scipy.optimize._shgo import SHGO
class TestShgoSimplicialTestFunctions:
    """
    Global optimisation tests with Simplicial sampling:
    """

    def test_f1_1_simplicial(self):
        """Multivariate test function 1:
        x[0]**2 + x[1]**2 with bounds=[(-1, 6), (-1, 6)]"""
        run_test(test1_1, n=1, sampling_method='simplicial')

    def test_f1_2_simplicial(self):
        """Multivariate test function 1:
        x[0]**2 + x[1]**2 with bounds=[(0, 1), (0, 1)]"""
        run_test(test1_2, n=1, sampling_method='simplicial')

    def test_f1_3_simplicial(self):
        """Multivariate test function 1: x[0]**2 + x[1]**2
        with bounds=[(None, None),(None, None)]"""
        run_test(test1_3, n=5, sampling_method='simplicial')

    def test_f2_1_simplicial(self):
        """Univariate test function on
        f(x) = (x - 30) * sin(x) with bounds=[(0, 60)]"""
        options = {'minimize_every_iter': False}
        run_test(test2_1, n=200, iters=7, options=options, sampling_method='simplicial')

    def test_f2_2_simplicial(self):
        """Univariate test function on
        f(x) = (x - 30) * sin(x) bounds=[(0, 4.5)]"""
        run_test(test2_2, n=1, sampling_method='simplicial')

    def test_f3_simplicial(self):
        """NLP: Hock and Schittkowski problem 18"""
        run_test(test3_1, n=1, sampling_method='simplicial')

    @pytest.mark.slow
    def test_f4_simplicial(self):
        """NLP: (High dimensional) Hock and Schittkowski 11 problem (HS11)"""
        run_test(test4_1, n=1, sampling_method='simplicial')

    def test_lj_symmetry_old(self):
        """LJ: Symmetry-constrained test function"""
        options = {'symmetry': True, 'disp': True}
        args = (6,)
        run_test(testLJ, args=args, n=300, options=options, iters=1, sampling_method='simplicial')

    def test_f5_1_lj_symmetry(self):
        """LJ: Symmetry constrained test function"""
        options = {'symmetry': [0] * 6, 'disp': True}
        args = (6,)
        run_test(testLJ, args=args, n=300, options=options, iters=1, sampling_method='simplicial')

    def test_f5_2_cons_symmetry(self):
        """Symmetry constrained test function"""
        options = {'symmetry': [0, 0], 'disp': True}
        run_test(test1_1, n=200, options=options, iters=1, sampling_method='simplicial')

    def test_f5_3_cons_symmetry(self):
        """Assymmetrically constrained test function"""
        options = {'symmetry': [0, 0, 0, 3], 'disp': True}
        run_test(test_s, n=10000, options=options, iters=1, sampling_method='simplicial')

    @pytest.mark.skip('Not a test')
    def test_f0_min_variance(self):
        """Return a minimum on a perfectly symmetric problem, based on
            gh10429"""
        avg = 0.5
        cons = {'type': 'eq', 'fun': lambda x: numpy.mean(x) - avg}
        res = shgo(numpy.var, bounds=6 * [(0, 1)], constraints=cons)
        assert res.success
        assert_allclose(res.fun, 0, atol=1e-15)
        assert_allclose(res.x, 0.5)

    @pytest.mark.skip('Not a test')
    def test_f0_min_variance_1D(self):
        """Return a minimum on a perfectly symmetric 1D problem, based on
            gh10538"""

        def fun(x):
            return x * (x - 1.0) * (x - 0.5)
        bounds = [(0, 1)]
        res = shgo(fun, bounds=bounds)
        ref = minimize_scalar(fun, bounds=bounds[0])
        assert res.success
        assert_allclose(res.fun, ref.fun)
        assert_allclose(res.x, ref.x, rtol=1e-06)