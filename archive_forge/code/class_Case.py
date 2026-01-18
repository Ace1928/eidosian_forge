import itertools
import platform
import sys
import pytest
import numpy as np
from numpy.testing import assert_array_equal, assert_allclose
from numpy import zeros, arange, array, ones, eye, iscomplexobj
from numpy.linalg import norm
from scipy.sparse import spdiags, csr_matrix, kronsum
from scipy.sparse.linalg import LinearOperator, aslinearoperator
from scipy.sparse.linalg._isolve import (bicg, bicgstab, cg, cgs,
class Case:

    def __init__(self, name, A, b=None, skip=None, nonconvergence=None):
        self.name = name
        self.A = A
        if b is None:
            self.b = arange(A.shape[0], dtype=float)
        else:
            self.b = b
        if skip is None:
            self.skip = []
        else:
            self.skip = skip
        if nonconvergence is None:
            self.nonconvergence = []
        else:
            self.nonconvergence = nonconvergence