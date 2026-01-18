import math
import re
import textwrap
import operator
import numpy as np
import unittest
from numba import jit, njit
from numba.core import types
from numba.core.errors import TypingError
from numba.core.types.functions import _header_lead
from numba.tests.support import TestCase
class TestArgumentTypingError(unittest.TestCase):
    """
    Test diagnostics of typing errors caused by argument inference failure.
    """

    def test_unsupported_array_dtype(self):
        cfunc = jit(nopython=True)(nop)
        a = np.ones(3)
        a = a.astype(a.dtype.newbyteorder())
        with self.assertRaises(TypingError) as raises:
            cfunc(1, a, a)
        expected = f'Unsupported array dtype: {a.dtype}'
        self.assertIn(expected, str(raises.exception))

    def test_unsupported_type(self):
        cfunc = jit(nopython=True)(nop)
        foo = Foo()
        with self.assertRaises(TypingError) as raises:
            cfunc(1, foo, 1)
        expected = re.compile("This error may have been caused by the following argument\\(s\\):\\n- argument 1:.*Cannot determine Numba type of <class 'numba.tests.test_typingerror.Foo'>")
        self.assertTrue(expected.search(str(raises.exception)) is not None)