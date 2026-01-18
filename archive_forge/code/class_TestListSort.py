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
class TestListSort(MemoryLeakMixin, TestCase):

    def setUp(self):
        super(TestListSort, self).setUp()
        np.random.seed(0)

    def make(self, ctor, data):
        lst = ctor()
        lst.extend(data)
        return lst

    def make_both(self, data):
        return {'py': self.make(list, data), 'nb': self.make(List, data)}

    def test_sort_no_args(self):

        def udt(lst):
            lst.sort()
            return lst
        for nelem in [13, 29, 127]:
            my_lists = self.make_both(np.random.randint(0, nelem, nelem))
            self.assertEqual(list(udt(my_lists['nb'])), udt(my_lists['py']))

    def test_sort_all_args(self):

        def udt(lst, key, reverse):
            lst.sort(key=key, reverse=reverse)
            return lst
        possible_keys = [lambda x: -x, lambda x: 1 / (1 + x), lambda x: (x, -x), lambda x: x]
        possible_reverse = [True, False]
        for key, reverse in product(possible_keys, possible_reverse):
            my_lists = self.make_both(np.random.randint(0, 100, 23))
            msg = 'case for key={} reverse={}'.format(key, reverse)
            self.assertEqual(list(udt(my_lists['nb'], key=key, reverse=reverse)), udt(my_lists['py'], key=key, reverse=reverse), msg=msg)

    def test_sort_dispatcher_key(self):

        def udt(lst, key):
            lst.sort(key=key)
            return lst
        my_lists = self.make_both(np.random.randint(0, 100, 31))
        py_key = lambda x: x + 1
        nb_key = njit(lambda x: x + 1)
        self.assertEqual(list(udt(my_lists['nb'], key=nb_key)), udt(my_lists['py'], key=py_key))
        self.assertEqual(list(udt(my_lists['nb'], key=nb_key)), list(udt(my_lists['nb'], key=py_key)))

    def test_sort_in_jit_w_lambda_key(self):

        @njit
        def udt(lst):
            lst.sort(key=lambda x: -x)
            return lst
        lst = self.make(List, np.random.randint(0, 100, 31))
        self.assertEqual(udt(lst), udt.py_func(lst))

    def test_sort_in_jit_w_global_key(self):

        @njit
        def keyfn(x):
            return -x

        @njit
        def udt(lst):
            lst.sort(key=keyfn)
            return lst
        lst = self.make(List, np.random.randint(0, 100, 31))
        self.assertEqual(udt(lst), udt.py_func(lst))

    def test_sort_on_arrays(self):

        @njit
        def foo(lst):
            lst.sort(key=lambda arr: np.sum(arr))
            return lst
        arrays = [np.random.random(3) for _ in range(10)]
        my_lists = self.make_both(arrays)
        self.assertEqual(list(foo(my_lists['nb'])), foo.py_func(my_lists['py']))