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
class TestTensorsolve:

    @pytest.mark.parametrize('a, axes', [(np.ones((4, 6, 8, 2)), None), (np.ones((3, 3, 2)), (0, 2))])
    def test_non_square_handling(self, a, axes):
        with assert_raises(LinAlgError):
            b = np.ones(a.shape[:2])
            linalg.tensorsolve(a, b, axes=axes)

    @pytest.mark.parametrize('shape', [(2, 3, 6), (3, 4, 4, 3), (0, 3, 3, 0)])
    def test_tensorsolve_result(self, shape):
        a = np.random.randn(*shape)
        b = np.ones(a.shape[:2])
        x = np.linalg.tensorsolve(a, b)
        assert_allclose(np.tensordot(a, x, axes=len(x.shape)), b)