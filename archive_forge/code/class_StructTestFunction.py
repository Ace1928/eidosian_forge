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
class StructTestFunction:

    def __init__(self, bounds, expected_x, expected_fun=None, expected_xl=None, expected_funl=None):
        self.bounds = bounds
        self.expected_x = expected_x
        self.expected_fun = expected_fun
        self.expected_xl = expected_xl
        self.expected_funl = expected_funl