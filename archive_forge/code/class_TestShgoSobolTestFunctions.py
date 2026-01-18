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
class TestShgoSobolTestFunctions:
    """
    Global optimisation tests with Sobol sampling:
    """

    def test_f1_1_sobol(self):
        """Multivariate test function 1:
        x[0]**2 + x[1]**2 with bounds=[(-1, 6), (-1, 6)]"""
        run_test(test1_1)

    def test_f1_2_sobol(self):
        """Multivariate test function 1:
         x[0]**2 + x[1]**2 with bounds=[(0, 1), (0, 1)]"""
        run_test(test1_2)

    def test_f1_3_sobol(self):
        """Multivariate test function 1:
        x[0]**2 + x[1]**2 with bounds=[(None, None),(None, None)]"""
        options = {'disp': True}
        run_test(test1_3, options=options)

    def test_f2_1_sobol(self):
        """Univariate test function on
        f(x) = (x - 30) * sin(x) with bounds=[(0, 60)]"""
        run_test(test2_1)

    def test_f2_2_sobol(self):
        """Univariate test function on
        f(x) = (x - 30) * sin(x) bounds=[(0, 4.5)]"""
        run_test(test2_2)

    def test_f3_sobol(self):
        """NLP: Hock and Schittkowski problem 18"""
        run_test(test3_1)

    @pytest.mark.slow
    def test_f4_sobol(self):
        """NLP: (High dimensional) Hock and Schittkowski 11 problem (HS11)"""
        options = {'infty_constraints': False}
        run_test(test4_1, n=990 * 2, options=options)

    def test_f5_1_sobol(self):
        """NLP: Eggholder, multimodal"""
        run_test(test5_1, n=60)

    def test_f5_2_sobol(self):
        """NLP: Eggholder, multimodal"""
        run_test(test5_1, n=60, iters=5)