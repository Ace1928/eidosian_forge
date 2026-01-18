import unittest
from functools import reduce
import numpy as np
from numba import njit, typeof, prange, pndindex
import numba.parfors.parfor
from numba.core import (
from numba.core.registry import cpu_target
from numba.tests.support import (TestCase, is_parfors_unsupported)
class TestConvertSetItemPass(BaseTest):
    sub_pass_class = numba.parfors.parfor.ConvertSetItemPass

    def test_setitem_full_slice(self):

        def test_impl():
            n = 10
            a = np.ones(n)
            a[:] = 7
            return a
        sub_pass = self.run_parfor_sub_pass(test_impl, ())
        self.assertEqual(len(sub_pass.rewritten), 1)
        [record] = sub_pass.rewritten
        self.assertEqual(record['reason'], 'slice')
        self.check_records(sub_pass.rewritten)
        self.run_parallel(test_impl)

    def test_setitem_slice_stop_bound(self):

        def test_impl():
            n = 10
            a = np.ones(n)
            a[:5] = 7
            return a
        sub_pass = self.run_parfor_sub_pass(test_impl, ())
        self.assertEqual(len(sub_pass.rewritten), 1)
        [record] = sub_pass.rewritten
        self.assertEqual(record['reason'], 'slice')
        self.check_records(sub_pass.rewritten)
        self.run_parallel(test_impl)

    def test_setitem_slice_start_bound(self):

        def test_impl():
            n = 10
            a = np.ones(n)
            a[4:] = 7
            return a
        sub_pass = self.run_parfor_sub_pass(test_impl, ())
        self.assertEqual(len(sub_pass.rewritten), 1)
        [record] = sub_pass.rewritten
        self.assertEqual(record['reason'], 'slice')
        self.check_records(sub_pass.rewritten)
        self.run_parallel(test_impl)

    def test_setitem_gather_if_scalar(self):

        def test_impl():
            n = 10
            a = np.ones(n)
            b = np.ones_like(a, dtype=np.bool_)
            a[b] = 7
            return a
        sub_pass = self.run_parfor_sub_pass(test_impl, ())
        self.assertEqual(len(sub_pass.rewritten), 1)
        [record] = sub_pass.rewritten
        self.assertEqual(record['reason'], 'masked_assign_broadcast_scalar')
        self.check_records(sub_pass.rewritten)
        self.run_parallel(test_impl)

    def test_setitem_gather_if_array(self):

        def test_impl():
            n = 10
            a = np.ones(n)
            b = np.ones_like(a, dtype=np.bool_)
            c = np.ones_like(a)
            a[b] = c[b]
            return a
        sub_pass = self.run_parfor_sub_pass(test_impl, ())
        self.assertEqual(len(sub_pass.rewritten), 1)
        [record] = sub_pass.rewritten
        self.assertEqual(record['reason'], 'masked_assign_array')
        self.check_records(sub_pass.rewritten)
        self.run_parallel(test_impl)