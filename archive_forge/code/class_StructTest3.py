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
class StructTest3(StructTestFunction):
    """
    Hock and Schittkowski 18 problem (HS18). Hoch and Schittkowski (1981)
    http://www.ai7.uni-bayreuth.de/test_problem_coll.pdf
    Minimize: f = 0.01 * (x_1)**2 + (x_2)**2

    Subject to: x_1 * x_2 - 25.0 >= 0,
                (x_1)**2 + (x_2)**2 - 25.0 >= 0,
                2 <= x_1 <= 50,
                0 <= x_2 <= 50.

    Approx. Answer:
        f([(250)**0.5 , (2.5)**0.5]) = 5.0


    """

    def f(self, x):
        return 0.01 * x[0] ** 2 + x[1] ** 2

    def g1(x):
        return x[0] * x[1] - 25.0

    def g2(x):
        return x[0] ** 2 + x[1] ** 2 - 25.0

    def g(x):
        return (x[0] * x[1] - 25.0, x[0] ** 2 + x[1] ** 2 - 25.0)
    __nlc = NonlinearConstraint(g, 0, np.inf)
    cons = (__nlc,)