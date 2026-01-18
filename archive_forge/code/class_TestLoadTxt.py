import sys
import gc
import gzip
import os
import threading
import time
import warnings
import io
import re
import pytest
from pathlib import Path
from tempfile import NamedTemporaryFile
from io import BytesIO, StringIO
from datetime import datetime
import locale
from multiprocessing import Value, get_context
from ctypes import c_bool
import numpy as np
import numpy.ma as ma
from numpy.lib._iotools import ConverterError, ConversionWarning
from numpy.compat import asbytes
from numpy.ma.testutils import assert_equal
from numpy.testing import (
from numpy.testing._private.utils import requires_memory
class TestLoadTxt(LoadTxtBase):
    loadfunc = staticmethod(np.loadtxt)

    def setup_method(self):
        self.orig_chunk = np.lib.npyio._loadtxt_chunksize
        np.lib.npyio._loadtxt_chunksize = 1

    def teardown_method(self):
        np.lib.npyio._loadtxt_chunksize = self.orig_chunk

    def test_record(self):
        c = TextIO()
        c.write('1 2\n3 4')
        c.seek(0)
        x = np.loadtxt(c, dtype=[('x', np.int32), ('y', np.int32)])
        a = np.array([(1, 2), (3, 4)], dtype=[('x', 'i4'), ('y', 'i4')])
        assert_array_equal(x, a)
        d = TextIO()
        d.write('M 64 75.0\nF 25 60.0')
        d.seek(0)
        mydescriptor = {'names': ('gender', 'age', 'weight'), 'formats': ('S1', 'i4', 'f4')}
        b = np.array([('M', 64.0, 75.0), ('F', 25.0, 60.0)], dtype=mydescriptor)
        y = np.loadtxt(d, dtype=mydescriptor)
        assert_array_equal(y, b)

    def test_array(self):
        c = TextIO()
        c.write('1 2\n3 4')
        c.seek(0)
        x = np.loadtxt(c, dtype=int)
        a = np.array([[1, 2], [3, 4]], int)
        assert_array_equal(x, a)
        c.seek(0)
        x = np.loadtxt(c, dtype=float)
        a = np.array([[1, 2], [3, 4]], float)
        assert_array_equal(x, a)

    def test_1D(self):
        c = TextIO()
        c.write('1\n2\n3\n4\n')
        c.seek(0)
        x = np.loadtxt(c, dtype=int)
        a = np.array([1, 2, 3, 4], int)
        assert_array_equal(x, a)
        c = TextIO()
        c.write('1,2,3,4\n')
        c.seek(0)
        x = np.loadtxt(c, dtype=int, delimiter=',')
        a = np.array([1, 2, 3, 4], int)
        assert_array_equal(x, a)

    def test_missing(self):
        c = TextIO()
        c.write('1,2,3,,5\n')
        c.seek(0)
        x = np.loadtxt(c, dtype=int, delimiter=',', converters={3: lambda s: int(s or -999)})
        a = np.array([1, 2, 3, -999, 5], int)
        assert_array_equal(x, a)

    def test_converters_with_usecols(self):
        c = TextIO()
        c.write('1,2,3,,5\n6,7,8,9,10\n')
        c.seek(0)
        x = np.loadtxt(c, dtype=int, delimiter=',', converters={3: lambda s: int(s or -999)}, usecols=(1, 3))
        a = np.array([[2, -999], [7, 9]], int)
        assert_array_equal(x, a)

    def test_comments_unicode(self):
        c = TextIO()
        c.write('# comment\n1,2,3,5\n')
        c.seek(0)
        x = np.loadtxt(c, dtype=int, delimiter=',', comments='#')
        a = np.array([1, 2, 3, 5], int)
        assert_array_equal(x, a)

    def test_comments_byte(self):
        c = TextIO()
        c.write('# comment\n1,2,3,5\n')
        c.seek(0)
        x = np.loadtxt(c, dtype=int, delimiter=',', comments=b'#')
        a = np.array([1, 2, 3, 5], int)
        assert_array_equal(x, a)

    def test_comments_multiple(self):
        c = TextIO()
        c.write('# comment\n1,2,3\n@ comment2\n4,5,6 // comment3')
        c.seek(0)
        x = np.loadtxt(c, dtype=int, delimiter=',', comments=['#', '@', '//'])
        a = np.array([[1, 2, 3], [4, 5, 6]], int)
        assert_array_equal(x, a)

    @pytest.mark.skipif(IS_PYPY and sys.implementation.version <= (7, 3, 8), reason='PyPy bug in error formatting')
    def test_comments_multi_chars(self):
        c = TextIO()
        c.write('/* comment\n1,2,3,5\n')
        c.seek(0)
        x = np.loadtxt(c, dtype=int, delimiter=',', comments='/*')
        a = np.array([1, 2, 3, 5], int)
        assert_array_equal(x, a)
        c = TextIO()
        c.write('*/ comment\n1,2,3,5\n')
        c.seek(0)
        assert_raises(ValueError, np.loadtxt, c, dtype=int, delimiter=',', comments='/*')

    def test_skiprows(self):
        c = TextIO()
        c.write('comment\n1,2,3,5\n')
        c.seek(0)
        x = np.loadtxt(c, dtype=int, delimiter=',', skiprows=1)
        a = np.array([1, 2, 3, 5], int)
        assert_array_equal(x, a)
        c = TextIO()
        c.write('# comment\n1,2,3,5\n')
        c.seek(0)
        x = np.loadtxt(c, dtype=int, delimiter=',', skiprows=1)
        a = np.array([1, 2, 3, 5], int)
        assert_array_equal(x, a)

    def test_usecols(self):
        a = np.array([[1, 2], [3, 4]], float)
        c = BytesIO()
        np.savetxt(c, a)
        c.seek(0)
        x = np.loadtxt(c, dtype=float, usecols=(1,))
        assert_array_equal(x, a[:, 1])
        a = np.array([[1, 2, 3], [3, 4, 5]], float)
        c = BytesIO()
        np.savetxt(c, a)
        c.seek(0)
        x = np.loadtxt(c, dtype=float, usecols=(1, 2))
        assert_array_equal(x, a[:, 1:])
        c.seek(0)
        x = np.loadtxt(c, dtype=float, usecols=np.array([1, 2]))
        assert_array_equal(x, a[:, 1:])
        for int_type in [int, np.int8, np.int16, np.int32, np.int64, np.uint8, np.uint16, np.uint32, np.uint64]:
            to_read = int_type(1)
            c.seek(0)
            x = np.loadtxt(c, dtype=float, usecols=to_read)
            assert_array_equal(x, a[:, 1])

        class CrazyInt:

            def __index__(self):
                return 1
        crazy_int = CrazyInt()
        c.seek(0)
        x = np.loadtxt(c, dtype=float, usecols=crazy_int)
        assert_array_equal(x, a[:, 1])
        c.seek(0)
        x = np.loadtxt(c, dtype=float, usecols=(crazy_int,))
        assert_array_equal(x, a[:, 1])
        data = 'JOE 70.1 25.3\n                BOB 60.5 27.9\n                '
        c = TextIO(data)
        names = ['stid', 'temp']
        dtypes = ['S4', 'f8']
        arr = np.loadtxt(c, usecols=(0, 2), dtype=list(zip(names, dtypes)))
        assert_equal(arr['stid'], [b'JOE', b'BOB'])
        assert_equal(arr['temp'], [25.3, 27.9])
        c.seek(0)
        bogus_idx = 1.5
        assert_raises_regex(TypeError, '^usecols must be.*%s' % type(bogus_idx).__name__, np.loadtxt, c, usecols=bogus_idx)
        assert_raises_regex(TypeError, '^usecols must be.*%s' % type(bogus_idx).__name__, np.loadtxt, c, usecols=[0, bogus_idx, 0])

    def test_bad_usecols(self):
        with pytest.raises(OverflowError):
            np.loadtxt(['1\n'], usecols=[2 ** 64], delimiter=',')
        with pytest.raises((ValueError, OverflowError)):
            np.loadtxt(['1\n'], usecols=[2 ** 62], delimiter=',')
        with pytest.raises(TypeError, match='If a structured dtype .*. But 1 usecols were given and the number of fields is 3.'):
            np.loadtxt(['1,1\n'], dtype='i,(2)i', usecols=[0], delimiter=',')

    def test_fancy_dtype(self):
        c = TextIO()
        c.write('1,2,3.0\n4,5,6.0\n')
        c.seek(0)
        dt = np.dtype([('x', int), ('y', [('t', int), ('s', float)])])
        x = np.loadtxt(c, dtype=dt, delimiter=',')
        a = np.array([(1, (2, 3.0)), (4, (5, 6.0))], dt)
        assert_array_equal(x, a)

    def test_shaped_dtype(self):
        c = TextIO('aaaa  1.0  8.0  1 2 3 4 5 6')
        dt = np.dtype([('name', 'S4'), ('x', float), ('y', float), ('block', int, (2, 3))])
        x = np.loadtxt(c, dtype=dt)
        a = np.array([('aaaa', 1.0, 8.0, [[1, 2, 3], [4, 5, 6]])], dtype=dt)
        assert_array_equal(x, a)

    def test_3d_shaped_dtype(self):
        c = TextIO('aaaa  1.0  8.0  1 2 3 4 5 6 7 8 9 10 11 12')
        dt = np.dtype([('name', 'S4'), ('x', float), ('y', float), ('block', int, (2, 2, 3))])
        x = np.loadtxt(c, dtype=dt)
        a = np.array([('aaaa', 1.0, 8.0, [[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]])], dtype=dt)
        assert_array_equal(x, a)

    def test_str_dtype(self):
        c = ['str1', 'str2']
        for dt in (str, np.bytes_):
            a = np.array(['str1', 'str2'], dtype=dt)
            x = np.loadtxt(c, dtype=dt)
            assert_array_equal(x, a)

    def test_empty_file(self):
        with pytest.warns(UserWarning, match='input contained no data'):
            c = TextIO()
            x = np.loadtxt(c)
            assert_equal(x.shape, (0,))
            x = np.loadtxt(c, dtype=np.int64)
            assert_equal(x.shape, (0,))
            assert_(x.dtype == np.int64)

    def test_unused_converter(self):
        c = TextIO()
        c.writelines(['1 21\n', '3 42\n'])
        c.seek(0)
        data = np.loadtxt(c, usecols=(1,), converters={0: lambda s: int(s, 16)})
        assert_array_equal(data, [21, 42])
        c.seek(0)
        data = np.loadtxt(c, usecols=(1,), converters={1: lambda s: int(s, 16)})
        assert_array_equal(data, [33, 66])

    def test_dtype_with_object(self):
        data = ' 1; 2001-01-01\n                   2; 2002-01-31 '
        ndtype = [('idx', int), ('code', object)]
        func = lambda s: strptime(s.strip(), '%Y-%m-%d')
        converters = {1: func}
        test = np.loadtxt(TextIO(data), delimiter=';', dtype=ndtype, converters=converters)
        control = np.array([(1, datetime(2001, 1, 1)), (2, datetime(2002, 1, 31))], dtype=ndtype)
        assert_equal(test, control)

    def test_uint64_type(self):
        tgt = (9223372043271415339, 9223372043271415853)
        c = TextIO()
        c.write('%s %s' % tgt)
        c.seek(0)
        res = np.loadtxt(c, dtype=np.uint64)
        assert_equal(res, tgt)

    def test_int64_type(self):
        tgt = (-9223372036854775807, 9223372036854775807)
        c = TextIO()
        c.write('%s %s' % tgt)
        c.seek(0)
        res = np.loadtxt(c, dtype=np.int64)
        assert_equal(res, tgt)

    def test_from_float_hex(self):
        tgt = np.logspace(-10, 10, 5).astype(np.float32)
        tgt = np.hstack((tgt, -tgt)).astype(float)
        inp = '\n'.join(map(float.hex, tgt))
        c = TextIO()
        c.write(inp)
        for dt in [float, np.float32]:
            c.seek(0)
            res = np.loadtxt(c, dtype=dt, converters=float.fromhex, encoding='latin1')
            assert_equal(res, tgt, err_msg='%s' % dt)

    @pytest.mark.skipif(IS_PYPY and sys.implementation.version <= (7, 3, 8), reason='PyPy bug in error formatting')
    def test_default_float_converter_no_default_hex_conversion(self):
        """
        Ensure that fromhex is only used for values with the correct prefix and
        is not called by default. Regression test related to gh-19598.
        """
        c = TextIO('a b c')
        with pytest.raises(ValueError, match=".*convert string 'a' to float64 at row 0, column 1"):
            np.loadtxt(c)

    @pytest.mark.skipif(IS_PYPY and sys.implementation.version <= (7, 3, 8), reason='PyPy bug in error formatting')
    def test_default_float_converter_exception(self):
        """
        Ensure that the exception message raised during failed floating point
        conversion is correct. Regression test related to gh-19598.
        """
        c = TextIO('qrs tuv')
        with pytest.raises(ValueError, match="could not convert string 'qrs' to float64"):
            np.loadtxt(c)

    def test_from_complex(self):
        tgt = (complex(1, 1), complex(1, -1))
        c = TextIO()
        c.write('%s %s' % tgt)
        c.seek(0)
        res = np.loadtxt(c, dtype=complex)
        assert_equal(res, tgt)

    def test_complex_misformatted(self):
        a = np.zeros((2, 2), dtype=np.complex128)
        re = np.pi
        im = np.e
        a[:] = re - 1j * im
        c = BytesIO()
        np.savetxt(c, a, fmt='%.16e')
        c.seek(0)
        txt = c.read()
        c.seek(0)
        txt_bad = txt.replace(b'e+00-', b'e00+-')
        assert_(txt_bad != txt)
        c.write(txt_bad)
        c.seek(0)
        res = np.loadtxt(c, dtype=complex)
        assert_equal(res, a)

    def test_universal_newline(self):
        with temppath() as name:
            with open(name, 'w') as f:
                f.write('1 21\r3 42\r')
            data = np.loadtxt(name)
        assert_array_equal(data, [[1, 21], [3, 42]])

    def test_empty_field_after_tab(self):
        c = TextIO()
        c.write('1 \t2 \t3\tstart \n4\t5\t6\t  \n7\t8\t9.5\t')
        c.seek(0)
        dt = {'names': ('x', 'y', 'z', 'comment'), 'formats': ('<i4', '<i4', '<f4', '|S8')}
        x = np.loadtxt(c, dtype=dt, delimiter='\t')
        a = np.array([b'start ', b'  ', b''])
        assert_array_equal(x['comment'], a)

    def test_unpack_structured(self):
        txt = TextIO('M 21 72\nF 35 58')
        dt = {'names': ('a', 'b', 'c'), 'formats': ('|S1', '<i4', '<f4')}
        a, b, c = np.loadtxt(txt, dtype=dt, unpack=True)
        assert_(a.dtype.str == '|S1')
        assert_(b.dtype.str == '<i4')
        assert_(c.dtype.str == '<f4')
        assert_array_equal(a, np.array([b'M', b'F']))
        assert_array_equal(b, np.array([21, 35]))
        assert_array_equal(c, np.array([72.0, 58.0]))

    def test_ndmin_keyword(self):
        c = TextIO()
        c.write('1,2,3\n4,5,6')
        c.seek(0)
        assert_raises(ValueError, np.loadtxt, c, ndmin=3)
        c.seek(0)
        assert_raises(ValueError, np.loadtxt, c, ndmin=1.5)
        c.seek(0)
        x = np.loadtxt(c, dtype=int, delimiter=',', ndmin=1)
        a = np.array([[1, 2, 3], [4, 5, 6]])
        assert_array_equal(x, a)
        d = TextIO()
        d.write('0,1,2')
        d.seek(0)
        x = np.loadtxt(d, dtype=int, delimiter=',', ndmin=2)
        assert_(x.shape == (1, 3))
        d.seek(0)
        x = np.loadtxt(d, dtype=int, delimiter=',', ndmin=1)
        assert_(x.shape == (3,))
        d.seek(0)
        x = np.loadtxt(d, dtype=int, delimiter=',', ndmin=0)
        assert_(x.shape == (3,))
        e = TextIO()
        e.write('0\n1\n2')
        e.seek(0)
        x = np.loadtxt(e, dtype=int, delimiter=',', ndmin=2)
        assert_(x.shape == (3, 1))
        e.seek(0)
        x = np.loadtxt(e, dtype=int, delimiter=',', ndmin=1)
        assert_(x.shape == (3,))
        e.seek(0)
        x = np.loadtxt(e, dtype=int, delimiter=',', ndmin=0)
        assert_(x.shape == (3,))
        with pytest.warns(UserWarning, match='input contained no data'):
            f = TextIO()
            assert_(np.loadtxt(f, ndmin=2).shape == (0, 1))
            assert_(np.loadtxt(f, ndmin=1).shape == (0,))

    def test_generator_source(self):

        def count():
            for i in range(10):
                yield ('%d' % i)
        res = np.loadtxt(count())
        assert_array_equal(res, np.arange(10))

    def test_bad_line(self):
        c = TextIO()
        c.write('1 2 3\n4 5 6\n2 3')
        c.seek(0)
        assert_raises_regex(ValueError, '3', np.loadtxt, c)

    def test_none_as_string(self):
        c = TextIO()
        c.write('100,foo,200\n300,None,400')
        c.seek(0)
        dt = np.dtype([('x', int), ('a', 'S10'), ('y', int)])
        np.loadtxt(c, delimiter=',', dtype=dt, comments=None)

    @pytest.mark.skipif(locale.getpreferredencoding() == 'ANSI_X3.4-1968', reason='Wrong preferred encoding')
    def test_binary_load(self):
        butf8 = b'5,6,7,\xc3\x95scarscar\r\n15,2,3,hello\r\n20,2,3,\xc3\x95scar\r\n'
        sutf8 = butf8.decode('UTF-8').replace('\r', '').splitlines()
        with temppath() as path:
            with open(path, 'wb') as f:
                f.write(butf8)
            with open(path, 'rb') as f:
                x = np.loadtxt(f, encoding='UTF-8', dtype=np.str_)
            assert_array_equal(x, sutf8)
            with open(path, 'rb') as f:
                x = np.loadtxt(f, encoding='UTF-8', dtype='S')
            x = [b'5,6,7,\xc3\x95scarscar', b'15,2,3,hello', b'20,2,3,\xc3\x95scar']
            assert_array_equal(x, np.array(x, dtype='S'))

    def test_max_rows(self):
        c = TextIO()
        c.write('1,2,3,5\n4,5,7,8\n2,1,4,5')
        c.seek(0)
        x = np.loadtxt(c, dtype=int, delimiter=',', max_rows=1)
        a = np.array([1, 2, 3, 5], int)
        assert_array_equal(x, a)

    def test_max_rows_with_skiprows(self):
        c = TextIO()
        c.write('comments\n1,2,3,5\n4,5,7,8\n2,1,4,5')
        c.seek(0)
        x = np.loadtxt(c, dtype=int, delimiter=',', skiprows=1, max_rows=1)
        a = np.array([1, 2, 3, 5], int)
        assert_array_equal(x, a)
        c = TextIO()
        c.write('comment\n1,2,3,5\n4,5,7,8\n2,1,4,5')
        c.seek(0)
        x = np.loadtxt(c, dtype=int, delimiter=',', skiprows=1, max_rows=2)
        a = np.array([[1, 2, 3, 5], [4, 5, 7, 8]], int)
        assert_array_equal(x, a)

    def test_max_rows_with_read_continuation(self):
        c = TextIO()
        c.write('1,2,3,5\n4,5,7,8\n2,1,4,5')
        c.seek(0)
        x = np.loadtxt(c, dtype=int, delimiter=',', max_rows=2)
        a = np.array([[1, 2, 3, 5], [4, 5, 7, 8]], int)
        assert_array_equal(x, a)
        x = np.loadtxt(c, dtype=int, delimiter=',')
        a = np.array([2, 1, 4, 5], int)
        assert_array_equal(x, a)

    def test_max_rows_larger(self):
        c = TextIO()
        c.write('comment\n1,2,3,5\n4,5,7,8\n2,1,4,5')
        c.seek(0)
        x = np.loadtxt(c, dtype=int, delimiter=',', skiprows=1, max_rows=6)
        a = np.array([[1, 2, 3, 5], [4, 5, 7, 8], [2, 1, 4, 5]], int)
        assert_array_equal(x, a)

    @pytest.mark.parametrize(['skip', 'data'], [(1, ['ignored\n', '1,2\n', '\n', '3,4\n']), (1, ['ignored', '1,2', '', '3,4']), (1, StringIO('ignored\n1,2\n\n3,4')), (0, ['-1,0\n', '1,2\n', '\n', '3,4\n']), (0, ['-1,0', '1,2', '', '3,4']), (0, StringIO('-1,0\n1,2\n\n3,4'))])
    def test_max_rows_empty_lines(self, skip, data):
        with pytest.warns(UserWarning, match=f'Input line 3.*max_rows={3 - skip}'):
            res = np.loadtxt(data, dtype=int, skiprows=skip, delimiter=',', max_rows=3 - skip)
            assert_array_equal(res, [[-1, 0], [1, 2], [3, 4]][skip:])
        if isinstance(data, StringIO):
            data.seek(0)
        with warnings.catch_warnings():
            warnings.simplefilter('error', UserWarning)
            with pytest.raises(UserWarning):
                np.loadtxt(data, dtype=int, skiprows=skip, delimiter=',', max_rows=3 - skip)