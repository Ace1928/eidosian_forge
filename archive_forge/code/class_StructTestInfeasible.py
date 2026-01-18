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
class StructTestInfeasible(StructTestFunction):
    """
    Test function with no feasible domain.
    """

    def f(self, x, *args):
        return x[0] ** 2 + x[1] ** 2

    def g1(x):
        return x[0] + x[1] - 1

    def g2(x):
        return -(x[0] + x[1] - 1)

    def g3(x):
        return -x[0] + x[1] - 1

    def g4(x):
        return -(-x[0] + x[1] - 1)
    g = (g1, g2, g3, g4)
    cons = wrap_constraints(g)