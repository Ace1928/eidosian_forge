import contextlib
import functools
import operator
import platform
import itertools
import sys
from scipy._lib import _pep440
import numpy as np
from numpy import (arange, zeros, array, dot, asarray,
import random
from numpy.testing import (assert_equal, assert_array_equal,
from pytest import raises as assert_raises
import scipy.linalg
import scipy.sparse as sparse
from scipy.sparse import (csc_matrix, csr_matrix, dok_matrix,
from scipy.sparse._sputils import (supported_dtypes, isscalarlike,
from scipy.sparse.linalg import splu, expm, inv
from scipy._lib.decorator import decorator
from scipy._lib._util import ComplexWarning
import pytest
def __arith_init(self):
    self.__A = array([[-1.5, 6.5, 0, 2.25, 0, 0], [3.125, -7.875, 0.625, 0, 0, 0], [0, 0, -0.125, 1.0, 0, 0], [0, 0, 8.375, 0, 0, 0]], 'float64')
    self.__B = array([[0.375, 0, 0, 0, -5, 2.5], [14.25, -3.75, 0, 0, -0.125, 0], [0, 7.25, 0, 0, 0, 0], [18.5, -0.0625, 0, 0, 0, 0]], 'complex128')
    self.__B.imag = array([[1.25, 0, 0, 0, 6, -3.875], [2.25, 4.125, 0, 0, 0, 2.75], [0, 4.125, 0, 0, 0, 0], [-0.0625, 0, 0, 0, 0, 0]], 'float64')
    assert_array_equal((self.__A * 16).astype('int32'), 16 * self.__A)
    assert_array_equal((self.__B.real * 16).astype('int32'), 16 * self.__B.real)
    assert_array_equal((self.__B.imag * 16).astype('int32'), 16 * self.__B.imag)
    self.__Asp = self.spcreator(self.__A)
    self.__Bsp = self.spcreator(self.__B)