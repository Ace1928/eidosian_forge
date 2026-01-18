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
class _TestNormBase:
    dt = None
    dec = None

    @staticmethod
    def check_dtype(x, res):
        if issubclass(x.dtype.type, np.inexact):
            assert_equal(res.dtype, x.real.dtype)
        else:
            assert_(issubclass(res.dtype.type, np.floating))