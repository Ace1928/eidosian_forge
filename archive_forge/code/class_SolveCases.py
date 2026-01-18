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
class SolveCases(LinalgSquareTestCase, LinalgGeneralizedSquareTestCase):

    def do(self, a, b, tags):
        x = linalg.solve(a, b)
        assert_almost_equal(b, dot_generalized(a, x))
        assert_(consistent_subclass(x, b))