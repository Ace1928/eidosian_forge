import numpy as np
import ctypes
from numba import jit, literal_unroll, njit, typeof
from numba.core import types
from numba.core.itanium_mangler import mangle_type
from numba.core.errors import TypingError
import unittest
from numba.np import numpy_support
from numba.tests.support import TestCase, skip_ppc64le_issue6465
def _fillData(self, arr):
    for i in range(arr.size):
        arr[i]['m'] = i
    arr[0]['n'] = 'abcde'
    arr[1]['n'] = 'xyz'
    arr[2]['n'] = 'u\x00v\x00\x00'