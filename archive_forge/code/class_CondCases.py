import os
import sys
import itertools
import traceback
import textwrap
import subprocess
import pytest
import numpy as np
from numpy import array, single, double, csingle, cdouble, dot, identity, matmul
from numpy.core import swapaxes
from numpy import multiply, atleast_2d, inf, asarray
from numpy import linalg
from numpy.linalg import matrix_power, norm, matrix_rank, multi_dot, LinAlgError
from numpy.linalg.linalg import _multi_dot_matrix_chain_order
from numpy.testing import (
class CondCases(LinalgSquareTestCase, LinalgGeneralizedSquareTestCase):

    def do(self, a, b, tags):
        c = asarray(a)
        if 'size-0' in tags:
            assert_raises(LinAlgError, linalg.cond, c)
            return
        s = linalg.svd(c, compute_uv=False)
        assert_almost_equal(linalg.cond(a), s[..., 0] / s[..., -1], single_decimal=5, double_decimal=11)
        assert_almost_equal(linalg.cond(a, 2), s[..., 0] / s[..., -1], single_decimal=5, double_decimal=11)
        assert_almost_equal(linalg.cond(a, -2), s[..., -1] / s[..., 0], single_decimal=5, double_decimal=11)
        cinv = np.linalg.inv(c)
        assert_almost_equal(linalg.cond(a, 1), abs(c).sum(-2).max(-1) * abs(cinv).sum(-2).max(-1), single_decimal=5, double_decimal=11)
        assert_almost_equal(linalg.cond(a, -1), abs(c).sum(-2).min(-1) * abs(cinv).sum(-2).min(-1), single_decimal=5, double_decimal=11)
        assert_almost_equal(linalg.cond(a, np.inf), abs(c).sum(-1).max(-1) * abs(cinv).sum(-1).max(-1), single_decimal=5, double_decimal=11)
        assert_almost_equal(linalg.cond(a, -np.inf), abs(c).sum(-1).min(-1) * abs(cinv).sum(-1).min(-1), single_decimal=5, double_decimal=11)
        assert_almost_equal(linalg.cond(a, 'fro'), np.sqrt((abs(c) ** 2).sum(-1).sum(-1) * (abs(cinv) ** 2).sum(-1).sum(-1)), single_decimal=5, double_decimal=11)