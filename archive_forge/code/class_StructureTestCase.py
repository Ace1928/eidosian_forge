import platform
from platform import architecture as _architecture
import struct
import sys
import unittest
from ctypes.test import need_symbol
from ctypes import (CDLL, Array, Structure, Union, POINTER, sizeof, byref, alignment,
from ctypes.util import find_library
from struct import calcsize
import _ctypes_test
from collections import namedtuple
from test import support
class StructureTestCase(unittest.TestCase):
    formats = {'c': c_char, 'b': c_byte, 'B': c_ubyte, 'h': c_short, 'H': c_ushort, 'i': c_int, 'I': c_uint, 'l': c_long, 'L': c_ulong, 'q': c_longlong, 'Q': c_ulonglong, 'f': c_float, 'd': c_double}

    def test_simple_structs(self):
        for code, tp in self.formats.items():

            class X(Structure):
                _fields_ = [('x', c_char), ('y', tp)]
            self.assertEqual((sizeof(X), code), (calcsize('c%c0%c' % (code, code)), code))

    def test_unions(self):
        for code, tp in self.formats.items():

            class X(Union):
                _fields_ = [('x', c_char), ('y', tp)]
            self.assertEqual((sizeof(X), code), (calcsize('%c' % code), code))

    def test_struct_alignment(self):

        class X(Structure):
            _fields_ = [('x', c_char * 3)]
        self.assertEqual(alignment(X), calcsize('s'))
        self.assertEqual(sizeof(X), calcsize('3s'))

        class Y(Structure):
            _fields_ = [('x', c_char * 3), ('y', c_int)]
        self.assertEqual(alignment(Y), alignment(c_int))
        self.assertEqual(sizeof(Y), calcsize('3si'))

        class SI(Structure):
            _fields_ = [('a', X), ('b', Y)]
        self.assertEqual(alignment(SI), max(alignment(Y), alignment(X)))
        self.assertEqual(sizeof(SI), calcsize('3s0i 3si 0i'))

        class IS(Structure):
            _fields_ = [('b', Y), ('a', X)]
        self.assertEqual(alignment(SI), max(alignment(X), alignment(Y)))
        self.assertEqual(sizeof(IS), calcsize('3si 3s 0i'))

        class XX(Structure):
            _fields_ = [('a', X), ('b', X)]
        self.assertEqual(alignment(XX), alignment(X))
        self.assertEqual(sizeof(XX), calcsize('3s 3s 0s'))

    def test_empty(self):

        class X(Structure):
            _fields_ = []

        class Y(Union):
            _fields_ = []
        self.assertTrue(alignment(X) == alignment(Y) == 1)
        self.assertTrue(sizeof(X) == sizeof(Y) == 0)

        class XX(Structure):
            _fields_ = [('a', X), ('b', X)]
        self.assertEqual(alignment(XX), 1)
        self.assertEqual(sizeof(XX), 0)

    def test_fields(self):

        class X(Structure):
            _fields_ = [('x', c_int), ('y', c_char)]
        self.assertEqual(X.x.offset, 0)
        self.assertEqual(X.x.size, sizeof(c_int))
        self.assertEqual(X.y.offset, sizeof(c_int))
        self.assertEqual(X.y.size, sizeof(c_char))
        self.assertRaises((TypeError, AttributeError), setattr, X.x, 'offset', 92)
        self.assertRaises((TypeError, AttributeError), setattr, X.x, 'size', 92)

        class X(Union):
            _fields_ = [('x', c_int), ('y', c_char)]
        self.assertEqual(X.x.offset, 0)
        self.assertEqual(X.x.size, sizeof(c_int))
        self.assertEqual(X.y.offset, 0)
        self.assertEqual(X.y.size, sizeof(c_char))
        self.assertRaises((TypeError, AttributeError), setattr, X.x, 'offset', 92)
        self.assertRaises((TypeError, AttributeError), setattr, X.x, 'size', 92)

    def test_packed(self):

        class X(Structure):
            _fields_ = [('a', c_byte), ('b', c_longlong)]
            _pack_ = 1
        self.assertEqual(sizeof(X), 9)
        self.assertEqual(X.b.offset, 1)

        class X(Structure):
            _fields_ = [('a', c_byte), ('b', c_longlong)]
            _pack_ = 2
        self.assertEqual(sizeof(X), 10)
        self.assertEqual(X.b.offset, 2)
        import struct
        longlong_size = struct.calcsize('q')
        longlong_align = struct.calcsize('bq') - longlong_size

        class X(Structure):
            _fields_ = [('a', c_byte), ('b', c_longlong)]
            _pack_ = 4
        self.assertEqual(sizeof(X), min(4, longlong_align) + longlong_size)
        self.assertEqual(X.b.offset, min(4, longlong_align))

        class X(Structure):
            _fields_ = [('a', c_byte), ('b', c_longlong)]
            _pack_ = 8
        self.assertEqual(sizeof(X), min(8, longlong_align) + longlong_size)
        self.assertEqual(X.b.offset, min(8, longlong_align))
        d = {'_fields_': [('a', 'b'), ('b', 'q')], '_pack_': -1}
        self.assertRaises(ValueError, type(Structure), 'X', (Structure,), d)

    @support.cpython_only
    def test_packed_c_limits(self):
        import _testcapi
        d = {'_fields_': [('a', c_byte)], '_pack_': _testcapi.INT_MAX + 1}
        self.assertRaises(ValueError, type(Structure), 'X', (Structure,), d)
        d = {'_fields_': [('a', c_byte)], '_pack_': _testcapi.UINT_MAX + 2}
        self.assertRaises(ValueError, type(Structure), 'X', (Structure,), d)

    def test_initializers(self):

        class Person(Structure):
            _fields_ = [('name', c_char * 6), ('age', c_int)]
        self.assertRaises(TypeError, Person, 42)
        self.assertRaises(ValueError, Person, b'asldkjaslkdjaslkdj')
        self.assertRaises(TypeError, Person, 'Name', 'HI')
        self.assertEqual(Person(b'12345', 5).name, b'12345')
        self.assertEqual(Person(b'123456', 5).name, b'123456')
        self.assertRaises(ValueError, Person, b'1234567', 5)

    def test_conflicting_initializers(self):

        class POINT(Structure):
            _fields_ = [('phi', c_float), ('rho', c_float)]
        self.assertRaisesRegex(TypeError, 'phi', POINT, 2, 3, phi=4)
        self.assertRaisesRegex(TypeError, 'rho', POINT, 2, 3, rho=4)
        self.assertRaises(TypeError, POINT, 2, 3, 4)

    def test_keyword_initializers(self):

        class POINT(Structure):
            _fields_ = [('x', c_int), ('y', c_int)]
        pt = POINT(1, 2)
        self.assertEqual((pt.x, pt.y), (1, 2))
        pt = POINT(y=2, x=1)
        self.assertEqual((pt.x, pt.y), (1, 2))

    def test_invalid_field_types(self):

        class POINT(Structure):
            pass
        self.assertRaises(TypeError, setattr, POINT, '_fields_', [('x', 1), ('y', 2)])

    def test_invalid_name(self):

        def declare_with_name(name):

            class S(Structure):
                _fields_ = [(name, c_int)]
        self.assertRaises(TypeError, declare_with_name, b'x')

    def test_intarray_fields(self):

        class SomeInts(Structure):
            _fields_ = [('a', c_int * 4)]
        self.assertEqual(SomeInts((1, 2)).a[:], [1, 2, 0, 0])
        self.assertEqual(SomeInts((1, 2)).a[:], [1, 2, 0, 0])
        self.assertEqual(SomeInts((1, 2)).a[::-1], [0, 0, 2, 1])
        self.assertEqual(SomeInts((1, 2)).a[::2], [1, 0])
        self.assertEqual(SomeInts((1, 2)).a[1:5:6], [2])
        self.assertEqual(SomeInts((1, 2)).a[6:4:-1], [])
        self.assertEqual(SomeInts((1, 2, 3, 4)).a[:], [1, 2, 3, 4])
        self.assertEqual(SomeInts((1, 2, 3, 4)).a[:], [1, 2, 3, 4])
        self.assertRaises(RuntimeError, SomeInts, (1, 2, 3, 4, 5))

    def test_nested_initializers(self):

        class Phone(Structure):
            _fields_ = [('areacode', c_char * 6), ('number', c_char * 12)]

        class Person(Structure):
            _fields_ = [('name', c_char * 12), ('phone', Phone), ('age', c_int)]
        p = Person(b'Someone', (b'1234', b'5678'), 5)
        self.assertEqual(p.name, b'Someone')
        self.assertEqual(p.phone.areacode, b'1234')
        self.assertEqual(p.phone.number, b'5678')
        self.assertEqual(p.age, 5)

    @need_symbol('c_wchar')
    def test_structures_with_wchar(self):

        class PersonW(Structure):
            _fields_ = [('name', c_wchar * 12), ('age', c_int)]
        p = PersonW('Someone é')
        self.assertEqual(p.name, 'Someone é')
        self.assertEqual(PersonW('1234567890').name, '1234567890')
        self.assertEqual(PersonW('12345678901').name, '12345678901')
        self.assertEqual(PersonW('123456789012').name, '123456789012')
        self.assertRaises(ValueError, PersonW, '1234567890123')

    def test_init_errors(self):

        class Phone(Structure):
            _fields_ = [('areacode', c_char * 6), ('number', c_char * 12)]

        class Person(Structure):
            _fields_ = [('name', c_char * 12), ('phone', Phone), ('age', c_int)]
        cls, msg = self.get_except(Person, b'Someone', (1, 2))
        self.assertEqual(cls, RuntimeError)
        self.assertEqual(msg, '(Phone) TypeError: expected bytes, int found')
        cls, msg = self.get_except(Person, b'Someone', (b'a', b'b', b'c'))
        self.assertEqual(cls, RuntimeError)
        self.assertEqual(msg, '(Phone) TypeError: too many initializers')

    def test_huge_field_name(self):

        def create_class(length):

            class S(Structure):
                _fields_ = [('x' * length, c_int)]
        for length in [10 ** i for i in range(0, 8)]:
            try:
                create_class(length)
            except MemoryError:
                pass

    def get_except(self, func, *args):
        try:
            func(*args)
        except Exception as detail:
            return (detail.__class__, str(detail))

    def test_abstract_class(self):

        class X(Structure):
            _abstract_ = 'something'
        cls, msg = self.get_except(eval, 'X()', locals())
        self.assertEqual((cls, msg), (TypeError, 'abstract class'))

    def test_methods(self):
        self.assertIn('in_dll', dir(type(Structure)))
        self.assertIn('from_address', dir(type(Structure)))
        self.assertIn('in_dll', dir(type(Structure)))

    def test_positional_args(self):

        class W(Structure):
            _fields_ = [('a', c_int), ('b', c_int)]

        class X(W):
            _fields_ = [('c', c_int)]

        class Y(X):
            pass

        class Z(Y):
            _fields_ = [('d', c_int), ('e', c_int), ('f', c_int)]
        z = Z(1, 2, 3, 4, 5, 6)
        self.assertEqual((z.a, z.b, z.c, z.d, z.e, z.f), (1, 2, 3, 4, 5, 6))
        z = Z(1)
        self.assertEqual((z.a, z.b, z.c, z.d, z.e, z.f), (1, 0, 0, 0, 0, 0))
        self.assertRaises(TypeError, lambda: Z(1, 2, 3, 4, 5, 6, 7))

    def test_pass_by_value(self):

        class Test(Structure):
            _fields_ = [('first', c_ulong), ('second', c_ulong), ('third', c_ulong)]
        s = Test()
        s.first = 3735928559
        s.second = 3405691582
        s.third = 195894762
        dll = CDLL(_ctypes_test.__file__)
        func = dll._testfunc_large_struct_update_value
        func.argtypes = (Test,)
        func.restype = None
        func(s)
        self.assertEqual(s.first, 3735928559)
        self.assertEqual(s.second, 3405691582)
        self.assertEqual(s.third, 195894762)

    def test_pass_by_value_finalizer(self):
        finalizer_calls = []

        class Test(Structure):
            _fields_ = [('first', c_ulong), ('second', c_ulong), ('third', c_ulong)]

            def __del__(self):
                finalizer_calls.append('called')
        s = Test(1, 2, 3)
        self.assertGreater(sizeof(s), sizeof(c_void_p))
        dll = CDLL(_ctypes_test.__file__)
        func = dll._testfunc_large_struct_update_value
        func.argtypes = (Test,)
        func.restype = None
        func(s)
        self.assertEqual(finalizer_calls, [])
        self.assertEqual(s.first, 1)
        self.assertEqual(s.second, 2)
        self.assertEqual(s.third, 3)
        s = None
        support.gc_collect()
        self.assertEqual(finalizer_calls, ['called'])

    def test_pass_by_value_in_register(self):

        class X(Structure):
            _fields_ = [('first', c_uint), ('second', c_uint)]
        s = X()
        s.first = 3735928559
        s.second = 3405691582
        dll = CDLL(_ctypes_test.__file__)
        func = dll._testfunc_reg_struct_update_value
        func.argtypes = (X,)
        func.restype = None
        func(s)
        self.assertEqual(s.first, 3735928559)
        self.assertEqual(s.second, 3405691582)
        got = X.in_dll(dll, 'last_tfrsuv_arg')
        self.assertEqual(s.first, got.first)
        self.assertEqual(s.second, got.second)

    def _test_issue18060(self, Vector):
        if sys.platform == 'win32':
            libm = CDLL(find_library('msvcrt.dll'))
        else:
            libm = CDLL(find_library('m'))
        libm.atan2.argtypes = [Vector]
        libm.atan2.restype = c_double
        arg = Vector(y=0.0, x=-1.0)
        self.assertAlmostEqual(libm.atan2(arg), 3.141592653589793)

    @unittest.skipIf(_architecture() == ('64bit', 'WindowsPE'), "can't test Windows x64 build")
    @unittest.skipUnless(sys.byteorder == 'little', "can't test on this platform")
    def test_issue18060_a(self):

        class Base(Structure):
            _fields_ = [('y', c_double), ('x', c_double)]

        class Mid(Base):
            pass
        Mid._fields_ = []

        class Vector(Mid):
            pass
        self._test_issue18060(Vector)

    @unittest.skipIf(_architecture() == ('64bit', 'WindowsPE'), "can't test Windows x64 build")
    @unittest.skipUnless(sys.byteorder == 'little', "can't test on this platform")
    def test_issue18060_b(self):

        class Base(Structure):
            _fields_ = [('y', c_double), ('x', c_double)]

        class Mid(Base):
            _fields_ = []

        class Vector(Mid):
            _fields_ = []
        self._test_issue18060(Vector)

    @unittest.skipIf(_architecture() == ('64bit', 'WindowsPE'), "can't test Windows x64 build")
    @unittest.skipUnless(sys.byteorder == 'little', "can't test on this platform")
    def test_issue18060_c(self):

        class Base(Structure):
            _fields_ = [('y', c_double)]

        class Mid(Base):
            _fields_ = []

        class Vector(Mid):
            _fields_ = [('x', c_double)]
        self._test_issue18060(Vector)

    def test_array_in_struct(self):
        dll = CDLL(_ctypes_test.__file__)

        class Test2(Structure):
            _fields_ = [('data', c_ubyte * 16)]

        class Test3AParent(Structure):
            _fields_ = [('data', c_float * 2)]

        class Test3A(Test3AParent):
            _fields_ = [('more_data', c_float * 2)]

        class Test3B(Structure):
            _fields_ = [('data', c_double * 2)]

        class Test3C(Structure):
            _fields_ = [('data', c_double * 4)]

        class Test3D(Structure):
            _fields_ = [('data', c_double * 8)]

        class Test3E(Structure):
            _fields_ = [('data', c_double * 9)]
        s = Test2()
        expected = 0
        for i in range(16):
            s.data[i] = i
            expected += i
        func = dll._testfunc_array_in_struct2
        func.restype = c_int
        func.argtypes = (Test2,)
        result = func(s)
        self.assertEqual(result, expected)
        for i in range(16):
            self.assertEqual(s.data[i], i)
        s = Test3A()
        s.data[0] = 3.14159
        s.data[1] = 2.71828
        s.more_data[0] = -3.0
        s.more_data[1] = -2.0
        expected = 3.14159 + 2.71828 - 3.0 - 2.0
        func = dll._testfunc_array_in_struct3A
        func.restype = c_double
        func.argtypes = (Test3A,)
        result = func(s)
        self.assertAlmostEqual(result, expected, places=6)
        self.assertAlmostEqual(s.data[0], 3.14159, places=6)
        self.assertAlmostEqual(s.data[1], 2.71828, places=6)
        self.assertAlmostEqual(s.more_data[0], -3.0, places=6)
        self.assertAlmostEqual(s.more_data[1], -2.0, places=6)
        StructCtype = namedtuple('StructCtype', ['cls', 'cfunc1', 'cfunc2', 'items'])
        structs_to_test = [StructCtype(Test3B, dll._testfunc_array_in_struct3B, dll._testfunc_array_in_struct3B_set_defaults, 2), StructCtype(Test3C, dll._testfunc_array_in_struct3C, dll._testfunc_array_in_struct3C_set_defaults, 4), StructCtype(Test3D, dll._testfunc_array_in_struct3D, dll._testfunc_array_in_struct3D_set_defaults, 8), StructCtype(Test3E, dll._testfunc_array_in_struct3E, dll._testfunc_array_in_struct3E_set_defaults, 9)]
        for sut in structs_to_test:
            s = sut.cls()
            expected = 0
            for i in range(sut.items):
                float_i = float(i)
                s.data[i] = float_i
                expected += float_i
            func = sut.cfunc1
            func.restype = c_double
            func.argtypes = (sut.cls,)
            result = func(s)
            self.assertEqual(result, expected)
            for i in range(sut.items):
                self.assertEqual(s.data[i], float(i))
            func = sut.cfunc2
            func.restype = sut.cls
            result = func()
            for i in range(sut.items):
                self.assertEqual(result.data[i], float(i + 1))

    def test_38368(self):

        class U(Union):
            _fields_ = [('f1', c_uint8 * 16), ('f2', c_uint16 * 8), ('f3', c_uint32 * 4)]
        u = U()
        u.f3[0] = 19088743
        u.f3[1] = 2309737967
        u.f3[2] = 1985229328
        u.f3[3] = 4275878552
        f1 = [u.f1[i] for i in range(16)]
        f2 = [u.f2[i] for i in range(8)]
        if sys.byteorder == 'little':
            self.assertEqual(f1, [103, 69, 35, 1, 239, 205, 171, 137, 16, 50, 84, 118, 152, 186, 220, 254])
            self.assertEqual(f2, [17767, 291, 52719, 35243, 12816, 30292, 47768, 65244])

    @unittest.skipIf(True, 'Test disabled for now - see bpo-16575/bpo-16576')
    def test_union_by_value(self):

        class Nested1(Structure):
            _fields_ = [('an_int', c_int), ('another_int', c_int)]

        class Test4(Union):
            _fields_ = [('a_long', c_long), ('a_struct', Nested1)]

        class Nested2(Structure):
            _fields_ = [('an_int', c_int), ('a_union', Test4)]

        class Test5(Structure):
            _fields_ = [('an_int', c_int), ('nested', Nested2), ('another_int', c_int)]
        test4 = Test4()
        dll = CDLL(_ctypes_test.__file__)
        with self.assertRaises(TypeError) as ctx:
            func = dll._testfunc_union_by_value1
            func.restype = c_long
            func.argtypes = (Test4,)
            result = func(test4)
        self.assertEqual(ctx.exception.args[0], 'item 1 in _argtypes_ passes a union by value, which is unsupported.')
        test5 = Test5()
        with self.assertRaises(TypeError) as ctx:
            func = dll._testfunc_union_by_value2
            func.restype = c_long
            func.argtypes = (Test5,)
            result = func(test5)
        self.assertEqual(ctx.exception.args[0], 'item 1 in _argtypes_ passes a union by value, which is unsupported.')
        test4.a_long = 12345
        func = dll._testfunc_union_by_reference1
        func.restype = c_long
        func.argtypes = (POINTER(Test4),)
        result = func(byref(test4))
        self.assertEqual(result, 12345)
        self.assertEqual(test4.a_long, 0)
        self.assertEqual(test4.a_struct.an_int, 0)
        self.assertEqual(test4.a_struct.another_int, 0)
        test4.a_struct.an_int = 305397760
        test4.a_struct.another_int = 22136
        func = dll._testfunc_union_by_reference2
        func.restype = c_long
        func.argtypes = (POINTER(Test4),)
        result = func(byref(test4))
        self.assertEqual(result, 305419896)
        self.assertEqual(test4.a_long, 0)
        self.assertEqual(test4.a_struct.an_int, 0)
        self.assertEqual(test4.a_struct.another_int, 0)
        test5.an_int = 301989888
        test5.nested.an_int = 3429888
        test5.another_int = 120
        func = dll._testfunc_union_by_reference3
        func.restype = c_long
        func.argtypes = (POINTER(Test5),)
        result = func(byref(test5))
        self.assertEqual(result, 305419896)
        self.assertEqual(test5.an_int, 0)
        self.assertEqual(test5.nested.an_int, 0)
        self.assertEqual(test5.another_int, 0)

    @unittest.skipIf(True, 'Test disabled for now - see bpo-16575/bpo-16576')
    def test_bitfield_by_value(self):

        class Test6(Structure):
            _fields_ = [('A', c_int, 1), ('B', c_int, 2), ('C', c_int, 3), ('D', c_int, 2)]
        test6 = Test6()
        test6.A = 1
        test6.B = 3
        test6.C = 7
        test6.D = 3
        dll = CDLL(_ctypes_test.__file__)
        with self.assertRaises(TypeError) as ctx:
            func = dll._testfunc_bitfield_by_value1
            func.restype = c_long
            func.argtypes = (Test6,)
            result = func(test6)
        self.assertEqual(ctx.exception.args[0], 'item 1 in _argtypes_ passes a struct/union with a bitfield by value, which is unsupported.')
        func = dll._testfunc_bitfield_by_reference1
        func.restype = c_long
        func.argtypes = (POINTER(Test6),)
        result = func(byref(test6))
        self.assertEqual(result, -4)
        self.assertEqual(test6.A, 0)
        self.assertEqual(test6.B, 0)
        self.assertEqual(test6.C, 0)
        self.assertEqual(test6.D, 0)

        class Test7(Structure):
            _fields_ = [('A', c_uint, 1), ('B', c_uint, 2), ('C', c_uint, 3), ('D', c_uint, 2)]
        test7 = Test7()
        test7.A = 1
        test7.B = 3
        test7.C = 7
        test7.D = 3
        func = dll._testfunc_bitfield_by_reference2
        func.restype = c_long
        func.argtypes = (POINTER(Test7),)
        result = func(byref(test7))
        self.assertEqual(result, 14)
        self.assertEqual(test7.A, 0)
        self.assertEqual(test7.B, 0)
        self.assertEqual(test7.C, 0)
        self.assertEqual(test7.D, 0)

        class Test8(Union):
            _fields_ = [('A', c_int, 1), ('B', c_int, 2), ('C', c_int, 3), ('D', c_int, 2)]
        test8 = Test8()
        with self.assertRaises(TypeError) as ctx:
            func = dll._testfunc_bitfield_by_value2
            func.restype = c_long
            func.argtypes = (Test8,)
            result = func(test8)
        self.assertEqual(ctx.exception.args[0], 'item 1 in _argtypes_ passes a union by value, which is unsupported.')