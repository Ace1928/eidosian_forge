import numpy as np
import ctypes
from numba import jit, literal_unroll, njit, typeof
from numba.core import types
from numba.core.itanium_mangler import mangle_type
from numba.core.errors import TypingError
import unittest
from numba.np import numpy_support
from numba.tests.support import TestCase, skip_ppc64le_issue6465
@skip_ppc64le_issue6465
class TestRecordDtypeWithCharSeq(TestCase):

    def _createSampleaArray(self):
        self.refsample1d = np.recarray(3, dtype=recordwithcharseq)
        self.nbsample1d = np.zeros(3, dtype=recordwithcharseq)

    def _fillData(self, arr):
        for i in range(arr.size):
            arr[i]['m'] = i
        arr[0]['n'] = 'abcde'
        arr[1]['n'] = 'xyz'
        arr[2]['n'] = 'u\x00v\x00\x00'

    def setUp(self):
        self._createSampleaArray()
        self._fillData(self.refsample1d)
        self._fillData(self.nbsample1d)

    def get_cfunc(self, pyfunc):
        rectype = numpy_support.from_dtype(recordwithcharseq)
        return njit((rectype[:], types.intp))(pyfunc)

    def test_return_charseq(self):
        pyfunc = get_charseq
        cfunc = self.get_cfunc(pyfunc)
        for i in range(self.refsample1d.size):
            expected = pyfunc(self.refsample1d, i)
            got = cfunc(self.nbsample1d, i)
            self.assertEqual(expected, got)

    def test_npm_argument_charseq(self):

        def pyfunc(arr, i):
            return arr[i].n
        identity = njit(lambda x: x)

        @jit(nopython=True)
        def cfunc(arr, i):
            return identity(arr[i].n)
        for i in range(self.refsample1d.size):
            expected = pyfunc(self.refsample1d, i)
            got = cfunc(self.nbsample1d, i)
            self.assertEqual(expected, got)

    def test_py_argument_charseq(self):
        pyfunc = set_charseq
        rectype = numpy_support.from_dtype(recordwithcharseq)
        sig = (rectype[::1], types.intp, rectype.typeof('n'))
        cfunc = njit(sig)(pyfunc).overloads[sig].entry_point
        for i in range(self.refsample1d.size):
            chars = '{0}'.format(hex(i + 10))
            pyfunc(self.refsample1d, i, chars)
            cfunc(self.nbsample1d, i, chars)
            np.testing.assert_equal(self.refsample1d, self.nbsample1d)

    def test_py_argument_char_seq_near_overflow(self):
        pyfunc = set_charseq
        rectype = numpy_support.from_dtype(recordwithcharseq)
        sig = (rectype[::1], types.intp, rectype.typeof('n'))
        cfunc = njit(sig)(pyfunc).overloads[sig].entry_point
        cs_near_overflow = 'abcde'
        self.assertEqual(len(cs_near_overflow), recordwithcharseq['n'].itemsize)
        cfunc(self.nbsample1d, 0, cs_near_overflow)
        self.assertEqual(self.nbsample1d[0]['n'].decode('ascii'), cs_near_overflow)
        np.testing.assert_equal(self.refsample1d[1:], self.nbsample1d[1:])

    def test_py_argument_char_seq_truncate(self):
        pyfunc = set_charseq
        rectype = numpy_support.from_dtype(recordwithcharseq)
        sig = (rectype[::1], types.intp, rectype.typeof('n'))
        cfunc = njit(sig)(pyfunc).overloads[sig].entry_point
        cs_overflowed = 'abcdef'
        pyfunc(self.refsample1d, 1, cs_overflowed)
        cfunc(self.nbsample1d, 1, cs_overflowed)
        np.testing.assert_equal(self.refsample1d, self.nbsample1d)
        self.assertEqual(self.refsample1d[1].n, cs_overflowed[:-1].encode('ascii'))

    def test_return_charseq_tuple(self):
        pyfunc = get_charseq_tuple
        cfunc = self.get_cfunc(pyfunc)
        for i in range(self.refsample1d.size):
            expected = pyfunc(self.refsample1d, i)
            got = cfunc(self.nbsample1d, i)
            self.assertEqual(expected, got)