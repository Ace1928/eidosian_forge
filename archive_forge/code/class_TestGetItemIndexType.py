import sys
import subprocess
from itertools import product
from textwrap import dedent
import numpy as np
from numba import config
from numba import njit
from numba import int32, float32, prange, uint8
from numba.core import types
from numba import typeof
from numba.typed import List, Dict
from numba.core.errors import TypingError
from numba.tests.support import (TestCase, MemoryLeakMixin, override_config,
from numba.core.unsafe.refcount import get_refcount
from numba.experimental import jitclass
class TestGetItemIndexType(MemoryLeakMixin, TestCase):

    def test_indexing_with_uint8(self):
        """ Test for reproducer at https://github.com/numba/numba/issues/7250
        """

        @njit
        def foo():
            l = List.empty_list(uint8)
            for i in range(129):
                l.append(uint8(i))
            a = uint8(128)
            return l[a]
        self.assertEqual(foo(), 128)