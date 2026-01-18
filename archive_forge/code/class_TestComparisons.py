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
class TestComparisons(MemoryLeakMixin, TestCase):

    def _cmp_dance(self, expected, pa, pb, na, nb):
        self.assertEqual(cmp.py_func(pa, pb), expected)
        py_got = cmp.py_func(na, nb)
        self.assertEqual(py_got, expected)
        jit_got = cmp(na, nb)
        self.assertEqual(jit_got, expected)

    def test_empty_vs_empty(self):
        pa, pb = ([], [])
        na, nb = (to_tl(pa), to_tl(pb))
        expected = (False, True, True, False, True, False)
        self._cmp_dance(expected, pa, pb, na, nb)

    def test_empty_vs_singleton(self):
        pa, pb = ([], [0])
        na, nb = (to_tl(pa), to_tl(pb))
        expected = (True, True, False, True, False, False)
        self._cmp_dance(expected, pa, pb, na, nb)

    def test_singleton_vs_empty(self):
        pa, pb = ([0], [])
        na, nb = (to_tl(pa), to_tl(pb))
        expected = (False, False, False, True, True, True)
        self._cmp_dance(expected, pa, pb, na, nb)

    def test_singleton_vs_singleton_equal(self):
        pa, pb = ([0], [0])
        na, nb = (to_tl(pa), to_tl(pb))
        expected = (False, True, True, False, True, False)
        self._cmp_dance(expected, pa, pb, na, nb)

    def test_singleton_vs_singleton_less_than(self):
        pa, pb = ([0], [1])
        na, nb = (to_tl(pa), to_tl(pb))
        expected = (True, True, False, True, False, False)
        self._cmp_dance(expected, pa, pb, na, nb)

    def test_singleton_vs_singleton_greater_than(self):
        pa, pb = ([1], [0])
        na, nb = (to_tl(pa), to_tl(pb))
        expected = (False, False, False, True, True, True)
        self._cmp_dance(expected, pa, pb, na, nb)

    def test_equal(self):
        pa, pb = ([1, 2, 3], [1, 2, 3])
        na, nb = (to_tl(pa), to_tl(pb))
        expected = (False, True, True, False, True, False)
        self._cmp_dance(expected, pa, pb, na, nb)

    def test_first_shorter(self):
        pa, pb = ([1, 2], [1, 2, 3])
        na, nb = (to_tl(pa), to_tl(pb))
        expected = (True, True, False, True, False, False)
        self._cmp_dance(expected, pa, pb, na, nb)

    def test_second_shorter(self):
        pa, pb = ([1, 2, 3], [1, 2])
        na, nb = (to_tl(pa), to_tl(pb))
        expected = (False, False, False, True, True, True)
        self._cmp_dance(expected, pa, pb, na, nb)

    def test_first_less_than(self):
        pa, pb = ([1, 2, 2], [1, 2, 3])
        na, nb = (to_tl(pa), to_tl(pb))
        expected = (True, True, False, True, False, False)
        self._cmp_dance(expected, pa, pb, na, nb)

    def test_first_greater_than(self):
        pa, pb = ([1, 2, 3], [1, 2, 2])
        na, nb = (to_tl(pa), to_tl(pb))
        expected = (False, False, False, True, True, True)
        self._cmp_dance(expected, pa, pb, na, nb)

    def test_equals_non_list(self):
        l = to_tl([1, 2, 3])
        self.assertFalse(any(cmp.py_func(l, 1)))
        self.assertFalse(any(cmp(l, 1)))