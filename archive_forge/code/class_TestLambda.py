import functools
import itertools
import operator
import platform
import sys
import numpy as np
from numpy import (array, isnan, r_, arange, finfo, pi, sin, cos, tan, exp,
import pytest
from pytest import raises as assert_raises
from numpy.testing import (assert_equal, assert_almost_equal,
from scipy import special
import scipy.special._ufuncs as cephes
from scipy.special import ellipe, ellipk, ellipkm1
from scipy.special import elliprc, elliprd, elliprf, elliprg, elliprj
from scipy.special import mathieu_odd_coef, mathieu_even_coef, stirling2
from scipy._lib.deprecation import _NoValue
from scipy._lib._util import np_long, np_ulong
from scipy.special._basic import _FACTORIALK_LIMITS_64BITS, \
from scipy.special._testutils import with_special_errors, \
import math
class TestLambda:

    def test_lmbda(self):
        lam = special.lmbda(1, 0.1)
        lamr = (array([special.jn(0, 0.1), 2 * special.jn(1, 0.1) / 0.1]), array([special.jvp(0, 0.1), -2 * special.jv(1, 0.1) / 0.01 + 2 * special.jvp(1, 0.1) / 0.1]))
        assert_array_almost_equal(lam, lamr, 8)