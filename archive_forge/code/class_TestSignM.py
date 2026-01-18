import random
import functools
import numpy as np
from numpy import array, identity, dot, sqrt
from numpy.testing import (assert_array_almost_equal, assert_allclose, assert_,
import pytest
import scipy.linalg
from scipy.linalg import (funm, signm, logm, sqrtm, fractional_matrix_power,
from scipy.linalg import _matfuncs_inv_ssq
import scipy.linalg._expm_frechet
from scipy.optimize import minimize
class TestSignM:

    def test_nils(self):
        a = array([[29.2, -24.2, 69.5, 49.8, 7.0], [-9.2, 5.2, -18.0, -16.8, -2.0], [-10.0, 6.0, -20.0, -18.0, -2.0], [-9.6, 9.6, -25.5, -15.4, -2.0], [9.8, -4.8, 18.0, 18.2, 2.0]])
        cr = array([[11.94933333, -2.24533333, 15.31733333, 21.65333333, -2.24533333], [-3.84266667, 0.49866667, -4.59066667, -7.18666667, 0.49866667], [-4.08, 0.56, -4.92, -7.6, 0.56], [-4.03466667, 1.04266667, -5.59866667, -7.02666667, 1.04266667], [4.15733333, -0.50133333, 4.90933333, 7.81333333, -0.50133333]])
        r = signm(a)
        assert_array_almost_equal(r, cr)

    def test_defective1(self):
        a = array([[0.0, 1, 0, 0], [1, 0, 1, 0], [0, 0, 0, 1], [0, 0, 1, 0]])
        signm(a, disp=False)

    def test_defective2(self):
        a = array(([29.2, -24.2, 69.5, 49.8, 7.0], [-9.2, 5.2, -18.0, -16.8, -2.0], [-10.0, 6.0, -20.0, -18.0, -2.0], [-9.6, 9.6, -25.5, -15.4, -2.0], [9.8, -4.8, 18.0, 18.2, 2.0]))
        signm(a, disp=False)

    def test_defective3(self):
        a = array([[-2.0, 25.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, -3.0, 10.0, 3.0, 3.0, 3.0, 0.0], [0.0, 0.0, 2.0, 15.0, 3.0, 3.0, 0.0], [0.0, 0.0, 0.0, 0.0, 15.0, 3.0, 0.0], [0.0, 0.0, 0.0, 0.0, 3.0, 10.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, -2.0, 25.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -3.0]])
        signm(a, disp=False)