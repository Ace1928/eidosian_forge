import sys
import warnings
import numpy as np
from numba import njit, literally
from numba import int32, int64, float32, float64
from numba import typeof
from numba.typed import Dict, dictobject, List
from numba.typed.typedobjectutils import _sentry_safe_cast
from numba.core.errors import TypingError
from numba.core import types
from numba.tests.support import (TestCase, MemoryLeakMixin, unittest,
from numba.experimental import jitclass
from numba.extending import overload
class TestDictRefctTypes(MemoryLeakMixin, TestCase):

    def test_str_key(self):

        @njit
        def foo():
            d = Dict.empty(key_type=types.unicode_type, value_type=types.int32)
            d['123'] = 123
            d['321'] = 321
            return d
        d = foo()
        self.assertEqual(d['123'], 123)
        self.assertEqual(d['321'], 321)
        expect = {'123': 123, '321': 321}
        self.assertEqual(dict(d), expect)
        d['123'] = 231
        expect['123'] = 231
        self.assertEqual(d['123'], 231)
        self.assertEqual(dict(d), expect)
        nelem = 100
        for i in range(nelem):
            d[str(i)] = i
            expect[str(i)] = i
        for i in range(nelem):
            self.assertEqual(d[str(i)], i)
        self.assertEqual(dict(d), expect)

    def test_str_val(self):

        @njit
        def foo():
            d = Dict.empty(key_type=types.int32, value_type=types.unicode_type)
            d[123] = '123'
            d[321] = '321'
            return d
        d = foo()
        self.assertEqual(d[123], '123')
        self.assertEqual(d[321], '321')
        expect = {123: '123', 321: '321'}
        self.assertEqual(dict(d), expect)
        d[123] = '231'
        expect[123] = '231'
        self.assertEqual(dict(d), expect)
        nelem = 1
        for i in range(nelem):
            d[i] = str(i)
            expect[i] = str(i)
        for i in range(nelem):
            self.assertEqual(d[i], str(i))
        self.assertEqual(dict(d), expect)

    def test_str_key_array_value(self):
        np.random.seed(123)
        d = Dict.empty(key_type=types.unicode_type, value_type=types.float64[:])
        expect = []
        expect.append(np.random.random(10))
        d['mass'] = expect[-1]
        expect.append(np.random.random(20))
        d['velocity'] = expect[-1]
        for i in range(100):
            expect.append(np.random.random(i))
            d[str(i)] = expect[-1]
        self.assertEqual(len(d), len(expect))
        self.assertPreciseEqual(d['mass'], expect[0])
        self.assertPreciseEqual(d['velocity'], expect[1])
        for got, exp in zip(d.values(), expect):
            self.assertPreciseEqual(got, exp)
        self.assertTrue('mass' in d)
        self.assertTrue('velocity' in d)
        del d['mass']
        self.assertFalse('mass' in d)
        del d['velocity']
        self.assertFalse('velocity' in d)
        del expect[0:2]
        for i in range(90):
            k, v = d.popitem()
            w = expect.pop()
            self.assertPreciseEqual(v, w)
        expect.append(np.random.random(10))
        d['last'] = expect[-1]
        for got, exp in zip(d.values(), expect):
            self.assertPreciseEqual(got, exp)

    def test_dict_of_dict_int_keyval(self):

        def inner_numba_dict():
            d = Dict.empty(key_type=types.intp, value_type=types.intp)
            return d
        d = Dict.empty(key_type=types.intp, value_type=types.DictType(types.intp, types.intp))

        def usecase(d, make_inner_dict):
            for i in range(100):
                mid = make_inner_dict()
                for j in range(i + 1):
                    mid[j] = j * 10000
                d[i] = mid
            return d
        got = usecase(d, inner_numba_dict)
        expect = usecase({}, dict)
        self.assertIsInstance(expect, dict)
        self.assertEqual(dict(got), expect)
        for where in [12, 3, 6, 8, 10]:
            del got[where]
            del expect[where]
            self.assertEqual(dict(got), expect)

    def test_dict_of_dict_npm(self):
        inner_dict_ty = types.DictType(types.intp, types.intp)

        @njit
        def inner_numba_dict():
            d = Dict.empty(key_type=types.intp, value_type=types.intp)
            return d

        @njit
        def foo(count):
            d = Dict.empty(key_type=types.intp, value_type=inner_dict_ty)
            for i in range(count):
                d[i] = inner_numba_dict()
                for j in range(i + 1):
                    d[i][j] = j
            return d
        d = foo(100)
        ct = 0
        for k, dd in d.items():
            ct += 1
            self.assertEqual(len(dd), k + 1)
            for kk, vv in dd.items():
                self.assertEqual(kk, vv)
        self.assertEqual(ct, 100)

    def test_delitem(self):
        d = Dict.empty(types.int64, types.unicode_type)
        d[1] = 'apple'

        @njit
        def foo(x, k):
            del x[1]
        foo(d, 1)
        self.assertEqual(len(d), 0)
        self.assertFalse(d)

    def test_getitem_return_type(self):
        d = Dict.empty(types.int64, types.int64[:])
        d[1] = np.arange(10, dtype=np.int64)

        @njit
        def foo(d):
            d[1] += 100
            return d[1]
        foo(d)
        retty = foo.nopython_signatures[0].return_type
        self.assertIsInstance(retty, types.Array)
        self.assertNotIsInstance(retty, types.Optional)
        self.assertPreciseEqual(d[1], np.arange(10, dtype=np.int64) + 100)

    def test_storage_model_mismatch(self):
        dct = Dict()
        ref = [('a', True, 'a'), ('b', False, 'b'), ('c', False, 'c')]
        for x in ref:
            dct[x] = x
        for i, x in enumerate(ref):
            self.assertEqual(dct[x], x)