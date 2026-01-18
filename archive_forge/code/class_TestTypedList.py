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
class TestTypedList(MemoryLeakMixin, TestCase):

    def test_basic(self):
        l = List.empty_list(int32)
        self.assertEqual(len(l), 0)
        l.append(0)
        self.assertEqual(len(l), 1)
        l.append(0)
        l.append(0)
        l[0] = 10
        l[1] = 11
        l[2] = 12
        self.assertEqual(l[0], 10)
        self.assertEqual(l[1], 11)
        self.assertEqual(l[2], 12)
        self.assertEqual(l[-3], 10)
        self.assertEqual(l[-2], 11)
        self.assertEqual(l[-1], 12)
        for i in l:
            pass
        self.assertTrue(10 in l)
        self.assertFalse(0 in l)
        l.append(12)
        self.assertEqual(l.count(0), 0)
        self.assertEqual(l.count(10), 1)
        self.assertEqual(l.count(12), 2)
        self.assertEqual(len(l), 4)
        self.assertEqual(l.pop(), 12)
        self.assertEqual(len(l), 3)
        self.assertEqual(l.pop(1), 11)
        self.assertEqual(len(l), 2)
        l.extend((100, 200, 300))
        self.assertEqual(len(l), 5)
        self.assertEqual(list(l), [10, 12, 100, 200, 300])
        l.insert(0, 0)
        self.assertEqual(list(l), [0, 10, 12, 100, 200, 300])
        l.insert(3, 13)
        self.assertEqual(list(l), [0, 10, 12, 13, 100, 200, 300])
        l.insert(100, 400)
        self.assertEqual(list(l), [0, 10, 12, 13, 100, 200, 300, 400])
        l.remove(0)
        l.remove(400)
        l.remove(13)
        self.assertEqual(list(l), [10, 12, 100, 200, 300])
        l.clear()
        self.assertEqual(len(l), 0)
        self.assertEqual(list(l), [])
        l.extend(tuple(range(10, 20)))
        l.reverse()
        self.assertEqual(list(l), list(range(10, 20))[::-1])
        new = l.copy()
        self.assertEqual(list(new), list(range(10, 20))[::-1])
        self.assertEqual(l, new)
        new[-1] = 42
        self.assertNotEqual(l, new)
        self.assertEqual(l.index(15), 4)

    def test_list_extend_refines_on_unicode_type(self):

        @njit
        def foo(string):
            l = List()
            l.extend(string)
            return l
        for func in (foo, foo.py_func):
            for string in ('a', 'abc', '\nabc\t'):
                self.assertEqual(list(func(string)), list(string))

    def test_unsigned_access(self):
        L = List.empty_list(int32)
        ui32_0 = types.uint32(0)
        ui32_1 = types.uint32(1)
        ui32_2 = types.uint32(2)
        L.append(types.uint32(10))
        L.append(types.uint32(11))
        L.append(types.uint32(12))
        self.assertEqual(len(L), 3)
        self.assertEqual(L[ui32_0], 10)
        self.assertEqual(L[ui32_1], 11)
        self.assertEqual(L[ui32_2], 12)
        L[ui32_0] = 123
        L[ui32_1] = 456
        L[ui32_2] = 789
        self.assertEqual(L[ui32_0], 123)
        self.assertEqual(L[ui32_1], 456)
        self.assertEqual(L[ui32_2], 789)
        ui32_123 = types.uint32(123)
        ui32_456 = types.uint32(456)
        ui32_789 = types.uint32(789)
        self.assertEqual(L.index(ui32_123), 0)
        self.assertEqual(L.index(ui32_456), 1)
        self.assertEqual(L.index(ui32_789), 2)
        L.__delitem__(ui32_2)
        del L[ui32_1]
        self.assertEqual(len(L), 1)
        self.assertEqual(L[ui32_0], 123)
        L.append(2)
        L.append(3)
        L.append(4)
        self.assertEqual(len(L), 4)
        self.assertEqual(L.pop(), 4)
        self.assertEqual(L.pop(ui32_2), 3)
        self.assertEqual(L.pop(ui32_1), 2)
        self.assertEqual(L.pop(ui32_0), 123)

    def test_dtype(self):
        L = List.empty_list(int32)
        self.assertEqual(L._dtype, int32)
        L = List.empty_list(float32)
        self.assertEqual(L._dtype, float32)

        @njit
        def foo():
            li, lf = (List(), List())
            li.append(int32(1))
            lf.append(float32(1.0))
            return (li._dtype, lf._dtype)
        self.assertEqual(foo(), (np.dtype('int32'), np.dtype('float32')))
        self.assertEqual(foo.py_func(), (int32, float32))

    def test_dtype_raises_exception_on_untyped_list(self):
        with self.assertRaises(RuntimeError) as raises:
            L = List()
            L._dtype
        self.assertIn('invalid operation on untyped list', str(raises.exception))

    @skip_parfors_unsupported
    def test_unsigned_prange(self):

        @njit(parallel=True)
        def foo(a):
            r = types.uint64(3)
            s = types.uint64(0)
            for i in prange(r):
                s = s + a[i]
            return s
        a = List.empty_list(types.uint64)
        a.append(types.uint64(12))
        a.append(types.uint64(1))
        a.append(types.uint64(7))
        self.assertEqual(foo(a), 20)

    def test_compiled(self):

        @njit
        def producer():
            l = List.empty_list(int32)
            l.append(23)
            return l

        @njit
        def consumer(l):
            return l[0]
        l = producer()
        val = consumer(l)
        self.assertEqual(val, 23)

    def test_getitem_slice(self):
        """ Test getitem using a slice.

        This tests suffers from combinatorial explosion, so we parametrize it
        and compare results against the regular list in a quasi fuzzing
        approach.

        """
        rl = list(range(10, 20))
        tl = List.empty_list(int32)
        for i in range(10, 20):
            tl.append(i)
        start_range = list(range(-20, 30))
        stop_range = list(range(-20, 30))
        step_range = [-5, -4, -3, -2, -1, 1, 2, 3, 4, 5]
        self.assertEqual(rl, list(tl))
        self.assertEqual(rl[:], list(tl[:]))
        for sa in start_range:
            self.assertEqual(rl[sa:], list(tl[sa:]))
        for so in stop_range:
            self.assertEqual(rl[:so], list(tl[:so]))
        for se in step_range:
            self.assertEqual(rl[::se], list(tl[::se]))
        for sa, so in product(start_range, stop_range):
            self.assertEqual(rl[sa:so], list(tl[sa:so]))
        for sa, se in product(start_range, step_range):
            self.assertEqual(rl[sa::se], list(tl[sa::se]))
        for so, se in product(stop_range, step_range):
            self.assertEqual(rl[:so:se], list(tl[:so:se]))
        for sa, so, se in product(start_range, stop_range, step_range):
            self.assertEqual(rl[sa:so:se], list(tl[sa:so:se]))

    def test_setitem_slice(self):
        """ Test setitem using a slice.

        This tests suffers from combinatorial explosion, so we parametrize it
        and compare results against the regular list in a quasi fuzzing
        approach.

        """

        def setup(start=10, stop=20):
            rl_ = list(range(start, stop))
            tl_ = List.empty_list(int32)
            for i in range(start, stop):
                tl_.append(i)
            self.assertEqual(rl_, list(tl_))
            return (rl_, tl_)
        rl, tl = setup()
        rl[:], tl[:] = (rl, tl)
        self.assertEqual(rl, list(tl))
        rl, tl = setup()
        rl[len(rl):], tl[len(tl):] = (rl, tl)
        self.assertEqual(rl, list(tl))
        rl, tl = setup()
        rl[:0], tl[:0] = (rl, tl)
        self.assertEqual(rl, list(tl))
        rl, tl = setup()
        rl[3:5], tl[3:5] = (rl[6:8], tl[6:8])
        self.assertEqual(rl, list(tl))
        rl, tl = setup()
        rl[3:5], tl[3:5] = (rl[6:9], tl[6:9])
        self.assertEqual(rl, list(tl))
        rl, tl = setup()
        rl[3:5], tl[3:5] = (rl[6:7], tl[6:7])
        self.assertEqual(rl, list(tl))
        rl, tl = setup()
        rl[len(rl):] = list(range(110, 120))
        tl[len(tl):] = to_tl(range(110, 120))
        self.assertEqual(rl, list(tl))
        rl, tl = setup(0, 0)
        rl[len(rl):] = list(range(110, 120))
        tl[len(tl):] = to_tl(range(110, 120))
        self.assertEqual(rl, list(tl))
        rl, tl = setup(0, 1)
        rl[len(rl):] = list(range(110, 120))
        tl[len(tl):] = to_tl(range(110, 120))
        self.assertEqual(rl, list(tl))
        rl, tl = setup()
        rl[:0], tl[:0] = (list(range(110, 120)), to_tl(range(110, 120)))
        self.assertEqual(rl, list(tl))
        rl, tl = setup(0, 0)
        rl[:0], tl[:0] = (list(range(110, 120)), to_tl(range(110, 120)))
        self.assertEqual(rl, list(tl))
        rl, tl = setup(0, 1)
        rl[:0], tl[:0] = (list(range(110, 120)), to_tl(range(110, 120)))
        self.assertEqual(rl, list(tl))
        rl, tl = setup()
        rl[1:3], tl[1:3] = ([100, 200], to_tl([100, 200]))
        self.assertEqual(rl, list(tl))
        rl, tl = setup()
        rl[1:3], tl[1:3] = ([100, 200, 300, 400], to_tl([100, 200, 300, 400]))
        self.assertEqual(rl, list(tl))
        rl, tl = setup()
        rl[1:3], tl[1:3] = ([100], to_tl([100]))
        self.assertEqual(rl, list(tl))
        rl, tl = setup()
        rl[1:3], tl[1:3] = ([], to_tl([]))
        self.assertEqual(rl, list(tl))
        rl, tl = setup()
        rl[:], tl[:] = ([], to_tl([]))
        self.assertEqual(rl, list(tl))
        rl, tl = setup()
        rl[::2], tl[::2] = ([100, 200, 300, 400, 500], to_tl([100, 200, 300, 400, 500]))
        self.assertEqual(rl, list(tl))
        rl, tl = setup()
        rl[::-2], tl[::-2] = ([100, 200, 300, 400, 500], to_tl([100, 200, 300, 400, 500]))
        self.assertEqual(rl, list(tl))
        rl, tl = setup()
        rl[::-1], tl[::-1] = (rl, tl)
        self.assertEqual(rl, list(tl))

    def test_setitem_slice_value_error(self):
        self.disable_leak_check()
        tl = List.empty_list(int32)
        for i in range(10, 20):
            tl.append(i)
        assignment = List.empty_list(int32)
        for i in range(1, 4):
            assignment.append(i)
        with self.assertRaises(ValueError) as raises:
            tl[8:3:-1] = assignment
        self.assertIn('length mismatch for extended slice and sequence', str(raises.exception))

    def test_delitem_slice(self):
        """ Test delitem using a slice.

        This tests suffers from combinatorial explosion, so we parametrize it
        and compare results against the regular list in a quasi fuzzing
        approach.

        """

        def setup(start=10, stop=20):
            rl_ = list(range(start, stop))
            tl_ = List.empty_list(int32)
            for i in range(start, stop):
                tl_.append(i)
            self.assertEqual(rl_, list(tl_))
            return (rl_, tl_)
        start_range = list(range(-20, 30))
        stop_range = list(range(-20, 30))
        step_range = [-5, -4, -3, -2, -1, 1, 2, 3, 4, 5]
        rl, tl = setup()
        self.assertEqual(rl, list(tl))
        del rl[:]
        del tl[:]
        self.assertEqual(rl, list(tl))
        for sa in start_range:
            rl, tl = setup()
            del rl[sa:]
            del tl[sa:]
            self.assertEqual(rl, list(tl))
        for so in stop_range:
            rl, tl = setup()
            del rl[:so]
            del tl[:so]
            self.assertEqual(rl, list(tl))
        for se in step_range:
            rl, tl = setup()
            del rl[::se]
            del tl[::se]
            self.assertEqual(rl, list(tl))
        for sa, so in product(start_range, stop_range):
            rl, tl = setup()
            del rl[sa:so]
            del tl[sa:so]
            self.assertEqual(rl, list(tl))
        for sa, se in product(start_range, step_range):
            rl, tl = setup()
            del rl[sa::se]
            del tl[sa::se]
            self.assertEqual(rl, list(tl))
        for so, se in product(stop_range, step_range):
            rl, tl = setup()
            del rl[:so:se]
            del tl[:so:se]
            self.assertEqual(rl, list(tl))
        for sa, so, se in product(start_range, stop_range, step_range):
            rl, tl = setup()
            del rl[sa:so:se]
            del tl[sa:so:se]
            self.assertEqual(rl, list(tl))

    def test_list_create_no_jit_using_empty_list(self):
        with override_config('DISABLE_JIT', True):
            with forbid_codegen():
                l = List.empty_list(types.int32)
                self.assertEqual(type(l), list)

    def test_list_create_no_jit_using_List(self):
        with override_config('DISABLE_JIT', True):
            with forbid_codegen():
                l = List()
                self.assertEqual(type(l), list)

    def test_catch_global_typed_list(self):
        from numba.tests.typedlist_usecases import catch_global
        expected_message = "The use of a ListType[int32] type, assigned to variable 'global_typed_list' in globals, is not supported as globals are considered compile-time constants and there is no known way to compile a ListType[int32] type as a constant."
        with self.assertRaises(TypingError) as raises:
            njit(catch_global)()
        self.assertIn(expected_message, str(raises.exception))
        self.disable_leak_check()

    def test_repr(self):
        l = List()
        expected = 'ListType[Undefined]([])'
        self.assertEqual(expected, repr(l))
        l = List([int32(i) for i in (1, 2, 3)])
        expected = 'ListType[int32]([1, 2, 3])'
        self.assertEqual(expected, repr(l))

    def test_repr_long_list_ipython(self):
        args = ['-m', 'IPython', '--quiet', '--quick', '--no-banner', '--colors=NoColor', '-c']
        base_cmd = [sys.executable] + args
        try:
            subprocess.check_output(base_cmd + ['--version'])
        except subprocess.CalledProcessError as e:
            self.skipTest('ipython not found: return code %d' % e.returncode)
        repr_cmd = [' '.join(['import sys;', 'from numba.typed import List;', 'res = repr(List(range(1005)));', 'sys.stderr.write(res);'])]
        cmd = base_cmd + repr_cmd
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
        out, err = p.communicate()
        l = List(range(1005))
        expected = f'{typeof(l)}([{', '.join(map(str, l[:1000]))}, ...])'
        self.assertEqual(expected, err)

    def test_iter_mutates_self(self):
        self.disable_leak_check()

        @njit
        def foo(x):
            count = 0
            for i in x:
                if count > 1:
                    x.append(2.0)
                count += 1
        l = List()
        l.append(1.0)
        l.append(1.0)
        l.append(1.0)
        with self.assertRaises(RuntimeError) as raises:
            foo(l)
        msg = 'list was mutated during iteration'
        self.assertIn(msg, str(raises.exception))