import numpy as np
import ctypes
from numba import jit, literal_unroll, njit, typeof
from numba.core import types
from numba.core.itanium_mangler import mangle_type
from numba.core.errors import TypingError
import unittest
from numba.np import numpy_support
from numba.tests.support import TestCase, skip_ppc64le_issue6465
class TestRecordArrayGetItem(TestCase):

    def test_literal_variable(self):
        arr = np.array([1, 2], dtype=recordtype2)
        pyfunc = get_field1
        jitfunc = njit(pyfunc)
        self.assertEqual(pyfunc(arr[0]), jitfunc(arr[0]))

    def test_literal_unroll(self):
        arr = np.array([1, 2], dtype=recordtype2)
        pyfunc = get_field2
        jitfunc = njit(pyfunc)
        self.assertEqual(pyfunc(arr[0]), jitfunc(arr[0]))

    def test_literal_variable_global_tuple(self):
        arr = np.array([1, 2], dtype=recordtype2)
        pyfunc = get_field3
        jitfunc = njit(pyfunc)
        self.assertEqual(pyfunc(arr[0]), jitfunc(arr[0]))

    def test_literal_unroll_global_tuple(self):
        arr = np.array([1, 2], dtype=recordtype2)
        pyfunc = get_field4
        jitfunc = njit(pyfunc)
        self.assertEqual(pyfunc(arr[0]), jitfunc(arr[0]))

    def test_literal_unroll_free_var_tuple(self):
        fs = ('e', 'f')
        arr = np.array([1, 2], dtype=recordtype2)

        def get_field(rec):
            out = 0
            for f in literal_unroll(fs):
                out += rec[f]
            return out
        jitfunc = njit(get_field)
        self.assertEqual(get_field(arr[0]), jitfunc(arr[0]))

    def test_error_w_invalid_field(self):
        arr = np.array([1, 2], dtype=recordtype3)
        jitfunc = njit(get_field1)
        with self.assertRaises(TypingError) as raises:
            jitfunc(arr[0])
        self.assertIn("Field 'f' was not found in record with fields ('first', 'second')", str(raises.exception))

    def test_literal_unroll_dynamic_to_static_getitem_transform(self):
        keys = ('a', 'b', 'c')
        n = 5

        def pyfunc(rec):
            x = np.zeros((n,))
            for o in literal_unroll(keys):
                x += rec[o]
            return x
        dt = np.float64
        ldd = [np.arange(dt(n)) for x in keys]
        ldk = [(x, np.float64) for x in keys]
        rec = np.rec.fromarrays(ldd, dtype=ldk)
        expected = pyfunc(rec)
        got = njit(pyfunc)(rec)
        np.testing.assert_allclose(expected, got)