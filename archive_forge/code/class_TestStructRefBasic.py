import warnings
import numpy as np
from numba import typed, njit, errors, typeof
from numba.core import types
from numba.experimental import structref
from numba.extending import overload_method, overload_attribute
from numba.tests.support import (
class TestStructRefBasic(MemoryLeakMixin, TestCase):

    def test_structref_type(self):
        sr = types.StructRef([('a', types.int64)])
        self.assertEqual(sr.field_dict['a'], types.int64)
        sr = types.StructRef([('a', types.int64), ('b', types.float64)])
        self.assertEqual(sr.field_dict['a'], types.int64)
        self.assertEqual(sr.field_dict['b'], types.float64)
        with self.assertRaisesRegex(ValueError, 'expecting a str for field name'):
            types.StructRef([(1, types.int64)])
        with self.assertRaisesRegex(ValueError, 'expecting a Numba Type for field type'):
            types.StructRef([('a', 123)])

    def test_invalid_uses(self):
        with self.assertRaisesRegex(ValueError, 'cannot register'):
            structref.register(types.StructRef)
        with self.assertRaisesRegex(ValueError, 'cannot register'):
            structref.define_boxing(types.StructRef, MyStruct)

    def test_MySimplerStructType(self):
        vs = np.arange(10, dtype=np.intp)
        ctr = 13
        first_expected = vs + vs
        first_got = ctor_by_intrinsic(vs, ctr)
        self.assertNotIsInstance(first_got, MyStruct)
        self.assertPreciseEqual(first_expected, get_values(first_got))
        second_expected = first_expected + ctr * ctr
        second_got = compute_fields(first_got)
        self.assertPreciseEqual(second_expected, second_got)

    def test_MySimplerStructType_wrapper_has_no_attrs(self):
        vs = np.arange(10, dtype=np.intp)
        ctr = 13
        wrapper = ctor_by_intrinsic(vs, ctr)
        self.assertIsInstance(wrapper, structref.StructRefProxy)
        with self.assertRaisesRegex(AttributeError, 'values'):
            wrapper.values
        with self.assertRaisesRegex(AttributeError, 'counter'):
            wrapper.counter

    def test_MyStructType(self):
        vs = np.arange(10, dtype=np.float64)
        ctr = 11
        first_expected_arr = vs.copy()
        first_got = ctor_by_class(vs, ctr)
        self.assertIsInstance(first_got, MyStruct)
        self.assertPreciseEqual(first_expected_arr, first_got.values)
        second_expected = first_expected_arr + ctr
        second_got = compute_fields(first_got)
        self.assertPreciseEqual(second_expected, second_got)
        self.assertEqual(first_got.counter, ctr)

    def test_MyStructType_mixed_types(self):

        @njit
        def mixed_type(x, y, m, n):
            return (MyStruct(x, y), MyStruct(m, n))
        a, b = mixed_type(1, 2.3, 3.4j, (4,))
        self.assertEqual(a.values, 1)
        self.assertEqual(a.counter, 2.3)
        self.assertEqual(b.values, 3.4j)
        self.assertEqual(b.counter, (4,))

    def test_MyStructType_in_dict(self):
        td = typed.Dict()
        td['a'] = MyStruct(1, 2.3)
        self.assertEqual(td['a'].values, 1)
        self.assertEqual(td['a'].counter, 2.3)
        td['a'] = MyStruct(2, 3.3)
        self.assertEqual(td['a'].values, 2)
        self.assertEqual(td['a'].counter, 3.3)
        td['a'].values += 10
        self.assertEqual(td['a'].values, 12)
        self.assertEqual(td['a'].counter, 3.3)
        td['b'] = MyStruct(4, 5.6)

    def test_MyStructType_in_dict_mixed_type_error(self):
        self.disable_leak_check()
        td = typed.Dict()
        td['a'] = MyStruct(1, 2.3)
        self.assertEqual(td['a'].values, 1)
        self.assertEqual(td['a'].counter, 2.3)
        with self.assertRaisesRegex(errors.TypingError, 'Cannot cast numba.MyStructType'):
            td['b'] = MyStruct(2.3, 1)

    def test_MyStructType_hash_no_typeof_recursion(self):
        st = MyStruct(1, 2)
        typeof(st)
        self.assertEqual(hash(st), 3)