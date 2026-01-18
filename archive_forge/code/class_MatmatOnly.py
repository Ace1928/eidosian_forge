from functools import partial
from itertools import product
import operator
from pytest import raises as assert_raises, warns
from numpy.testing import assert_, assert_equal
import numpy as np
import scipy.sparse as sparse
import scipy.sparse.linalg._interface as interface
from scipy.sparse._sputils import matrix
class MatmatOnly(interface.LinearOperator):

    def __init__(self, A):
        super().__init__(A.dtype, A.shape)
        self.A = A

    def _matmat(self, x):
        return self.A.dot(x)