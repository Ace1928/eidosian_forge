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
class StructTest5(StructTestFunction):

    def f(self, x):
        return -(x[1] + 47.0) * numpy.sin(numpy.sqrt(abs(x[0] / 2.0 + (x[1] + 47.0)))) - x[0] * numpy.sin(numpy.sqrt(abs(x[0] - (x[1] + 47.0))))
    g = None
    cons = wrap_constraints(g)