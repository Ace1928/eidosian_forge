import unittest
import warnings
from contextlib import contextmanager
import numpy as np
import llvmlite.binding as llvm
from numba import njit, types
from numba.core.errors import NumbaInvalidConfigWarning
from numba.core.codegen import _parse_refprune_flags
from numba.tests.support import override_config, TestCase
class TestRefOpPruning(TestCase):
    _numba_parallel_test_ = False

    def check(self, func, *argtys, **prune_types):
        """
        Asserts the the func compiled with argument types "argtys" reports
        refop pruning statistics. The **prune_types** kwargs list each kind
        of pruning and whether the stat should be zero (False) or >0 (True).

        Note: The exact statistic varies across platform.

        NOTE: Tests using this `check` method need to run in subprocesses as
        `njit` sets up the module pass manager etc once and the overrides have
        no effect else.
        """
        with override_config('LLVM_REFPRUNE_PASS', '1'):
            cres = njit((*argtys,))(func).overloads[*argtys,]
        pstats = cres.metadata.get('prune_stats', None)
        self.assertIsNotNone(pstats)
        for k, v in prune_types.items():
            stat = getattr(pstats, k, None)
            self.assertIsNotNone(stat)
            msg = f'failed checking {k}'
            if v:
                self.assertGreater(stat, 0, msg=msg)
            else:
                self.assertEqual(stat, 0, msg=msg)

    @TestCase.run_test_in_subprocess
    def test_basic_block_1(self):

        def func(n):
            a = np.zeros(n)
            acc = 0
            if n > 4:
                b = a[1:]
                acc += b[1]
            else:
                c = a[:-1]
                acc += c[0]
            return acc
        self.check(func, types.intp, basicblock=True)

    @TestCase.run_test_in_subprocess
    def test_diamond_1(self):

        def func(n):
            a = np.ones(n)
            x = 0
            if n > 2:
                x = a.sum()
            return x + 1
        with set_refprune_flags('per_bb,diamond'):
            self.check(func, types.intp, basicblock=True, diamond=True, fanout=False, fanout_raise=False)

    @TestCase.run_test_in_subprocess
    def test_diamond_2(self):

        def func(n):
            con = []
            for i in range(n):
                con.append(np.arange(i))
            c = 0.0
            for arr in con:
                c += arr.sum() / (1 + arr.size)
            return c
        with set_refprune_flags('per_bb,diamond'):
            self.check(func, types.intp, basicblock=True, diamond=True, fanout=False, fanout_raise=False)

    @TestCase.run_test_in_subprocess
    def test_fanout_1(self):

        def func(n):
            a = np.zeros(n)
            b = np.zeros(n)
            x = (a, b)
            acc = 0.0
            for i in x:
                acc += i[0]
            return acc
        self.check(func, types.intp, basicblock=True, fanout=True)

    @TestCase.run_test_in_subprocess
    def test_fanout_2(self):

        def func(n):
            a = np.zeros(n)
            b = np.zeros(n)
            x = (a, b)
            for i in x:
                if n:
                    raise ValueError
            return x
        with set_refprune_flags('per_bb,fanout'):
            self.check(func, types.intp, basicblock=True, diamond=False, fanout=True, fanout_raise=False)

    @TestCase.run_test_in_subprocess
    def test_fanout_3(self):

        def func(n):
            ary = np.arange(n)
            c = 0
            for v in np.nditer(ary):
                c += v.item()
            return 1
        with set_refprune_flags('per_bb,fanout_raise'):
            self.check(func, types.intp, basicblock=True, diamond=False, fanout=False, fanout_raise=True)