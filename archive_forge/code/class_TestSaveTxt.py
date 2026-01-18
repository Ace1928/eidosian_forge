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
class TestSaveTxt:

    def test_array(self):
        a = np.array([[1, 2], [3, 4]], float)
        fmt = '%.18e'
        c = BytesIO()
        np.savetxt(c, a, fmt=fmt)
        c.seek(0)
        assert_equal(c.readlines(), [asbytes((fmt + ' ' + fmt + '\n') % (1, 2)), asbytes((fmt + ' ' + fmt + '\n') % (3, 4))])
        a = np.array([[1, 2], [3, 4]], int)
        c = BytesIO()
        np.savetxt(c, a, fmt='%d')
        c.seek(0)
        assert_equal(c.readlines(), [b'1 2\n', b'3 4\n'])

    def test_1D(self):
        a = np.array([1, 2, 3, 4], int)
        c = BytesIO()
        np.savetxt(c, a, fmt='%d')
        c.seek(0)
        lines = c.readlines()
        assert_equal(lines, [b'1\n', b'2\n', b'3\n', b'4\n'])

    def test_0D_3D(self):
        c = BytesIO()
        assert_raises(ValueError, np.savetxt, c, np.array(1))
        assert_raises(ValueError, np.savetxt, c, np.array([[[1], [2]]]))

    def test_structured(self):
        a = np.array([(1, 2), (3, 4)], dtype=[('x', 'i4'), ('y', 'i4')])
        c = BytesIO()
        np.savetxt(c, a, fmt='%d')
        c.seek(0)
        assert_equal(c.readlines(), [b'1 2\n', b'3 4\n'])

    def test_structured_padded(self):
        a = np.array([(1, 2, 3), (4, 5, 6)], dtype=[('foo', 'i4'), ('bar', 'i4'), ('baz', 'i4')])
        c = BytesIO()
        np.savetxt(c, a[['foo', 'baz']], fmt='%d')
        c.seek(0)
        assert_equal(c.readlines(), [b'1 3\n', b'4 6\n'])

    def test_multifield_view(self):
        a = np.ones(1, dtype=[('x', 'i4'), ('y', 'i4'), ('z', 'f4')])
        v = a[['x', 'z']]
        with temppath(suffix='.npy') as path:
            path = Path(path)
            np.save(path, v)
            data = np.load(path)
            assert_array_equal(data, v)

    def test_delimiter(self):
        a = np.array([[1.0, 2.0], [3.0, 4.0]])
        c = BytesIO()
        np.savetxt(c, a, delimiter=',', fmt='%d')
        c.seek(0)
        assert_equal(c.readlines(), [b'1,2\n', b'3,4\n'])

    def test_format(self):
        a = np.array([(1, 2), (3, 4)])
        c = BytesIO()
        np.savetxt(c, a, fmt=['%02d', '%3.1f'])
        c.seek(0)
        assert_equal(c.readlines(), [b'01 2.0\n', b'03 4.0\n'])
        c = BytesIO()
        np.savetxt(c, a, fmt='%02d : %3.1f')
        c.seek(0)
        lines = c.readlines()
        assert_equal(lines, [b'01 : 2.0\n', b'03 : 4.0\n'])
        c = BytesIO()
        np.savetxt(c, a, fmt='%02d : %3.1f', delimiter=',')
        c.seek(0)
        lines = c.readlines()
        assert_equal(lines, [b'01 : 2.0\n', b'03 : 4.0\n'])
        c = BytesIO()
        assert_raises(ValueError, np.savetxt, c, a, fmt=99)

    def test_header_footer(self):
        c = BytesIO()
        a = np.array([(1, 2), (3, 4)], dtype=int)
        test_header_footer = 'Test header / footer'
        np.savetxt(c, a, fmt='%1d', header=test_header_footer)
        c.seek(0)
        assert_equal(c.read(), asbytes('# ' + test_header_footer + '\n1 2\n3 4\n'))
        c = BytesIO()
        np.savetxt(c, a, fmt='%1d', footer=test_header_footer)
        c.seek(0)
        assert_equal(c.read(), asbytes('1 2\n3 4\n# ' + test_header_footer + '\n'))
        c = BytesIO()
        commentstr = '% '
        np.savetxt(c, a, fmt='%1d', header=test_header_footer, comments=commentstr)
        c.seek(0)
        assert_equal(c.read(), asbytes(commentstr + test_header_footer + '\n' + '1 2\n3 4\n'))
        c = BytesIO()
        commentstr = '% '
        np.savetxt(c, a, fmt='%1d', footer=test_header_footer, comments=commentstr)
        c.seek(0)
        assert_equal(c.read(), asbytes('1 2\n3 4\n' + commentstr + test_header_footer + '\n'))

    def test_file_roundtrip(self):
        with temppath() as name:
            a = np.array([(1, 2), (3, 4)])
            np.savetxt(name, a)
            b = np.loadtxt(name)
            assert_array_equal(a, b)

    def test_complex_arrays(self):
        ncols = 2
        nrows = 2
        a = np.zeros((ncols, nrows), dtype=np.complex128)
        re = np.pi
        im = np.e
        a[:] = re + 1j * im
        c = BytesIO()
        np.savetxt(c, a, fmt=' %+.3e')
        c.seek(0)
        lines = c.readlines()
        assert_equal(lines, [b' ( +3.142e+00+ +2.718e+00j)  ( +3.142e+00+ +2.718e+00j)\n', b' ( +3.142e+00+ +2.718e+00j)  ( +3.142e+00+ +2.718e+00j)\n'])
        c = BytesIO()
        np.savetxt(c, a, fmt='  %+.3e' * 2 * ncols)
        c.seek(0)
        lines = c.readlines()
        assert_equal(lines, [b'  +3.142e+00  +2.718e+00  +3.142e+00  +2.718e+00\n', b'  +3.142e+00  +2.718e+00  +3.142e+00  +2.718e+00\n'])
        c = BytesIO()
        np.savetxt(c, a, fmt=['(%.3e%+.3ej)'] * ncols)
        c.seek(0)
        lines = c.readlines()
        assert_equal(lines, [b'(3.142e+00+2.718e+00j) (3.142e+00+2.718e+00j)\n', b'(3.142e+00+2.718e+00j) (3.142e+00+2.718e+00j)\n'])

    def test_complex_negative_exponent(self):
        ncols = 2
        nrows = 2
        a = np.zeros((ncols, nrows), dtype=np.complex128)
        re = np.pi
        im = np.e
        a[:] = re - 1j * im
        c = BytesIO()
        np.savetxt(c, a, fmt='%.3e')
        c.seek(0)
        lines = c.readlines()
        assert_equal(lines, [b' (3.142e+00-2.718e+00j)  (3.142e+00-2.718e+00j)\n', b' (3.142e+00-2.718e+00j)  (3.142e+00-2.718e+00j)\n'])

    def test_custom_writer(self):

        class CustomWriter(list):

            def write(self, text):
                self.extend(text.split(b'\n'))
        w = CustomWriter()
        a = np.array([(1, 2), (3, 4)])
        np.savetxt(w, a)
        b = np.loadtxt(w)
        assert_array_equal(a, b)

    def test_unicode(self):
        utf8 = b'\xcf\x96'.decode('UTF-8')
        a = np.array([utf8], dtype=np.str_)
        with tempdir() as tmpdir:
            np.savetxt(os.path.join(tmpdir, 'test.csv'), a, fmt=['%s'], encoding='UTF-8')

    def test_unicode_roundtrip(self):
        utf8 = b'\xcf\x96'.decode('UTF-8')
        a = np.array([utf8], dtype=np.str_)
        suffixes = ['', '.gz']
        if HAS_BZ2:
            suffixes.append('.bz2')
        if HAS_LZMA:
            suffixes.extend(['.xz', '.lzma'])
        with tempdir() as tmpdir:
            for suffix in suffixes:
                np.savetxt(os.path.join(tmpdir, 'test.csv' + suffix), a, fmt=['%s'], encoding='UTF-16-LE')
                b = np.loadtxt(os.path.join(tmpdir, 'test.csv' + suffix), encoding='UTF-16-LE', dtype=np.str_)
                assert_array_equal(a, b)

    def test_unicode_bytestream(self):
        utf8 = b'\xcf\x96'.decode('UTF-8')
        a = np.array([utf8], dtype=np.str_)
        s = BytesIO()
        np.savetxt(s, a, fmt=['%s'], encoding='UTF-8')
        s.seek(0)
        assert_equal(s.read().decode('UTF-8'), utf8 + '\n')

    def test_unicode_stringstream(self):
        utf8 = b'\xcf\x96'.decode('UTF-8')
        a = np.array([utf8], dtype=np.str_)
        s = StringIO()
        np.savetxt(s, a, fmt=['%s'], encoding='UTF-8')
        s.seek(0)
        assert_equal(s.read(), utf8 + '\n')

    @pytest.mark.parametrize('fmt', ['%f', b'%f'])
    @pytest.mark.parametrize('iotype', [StringIO, BytesIO])
    def test_unicode_and_bytes_fmt(self, fmt, iotype):
        a = np.array([1.0])
        s = iotype()
        np.savetxt(s, a, fmt=fmt)
        s.seek(0)
        if iotype is StringIO:
            assert_equal(s.read(), '%f\n' % 1.0)
        else:
            assert_equal(s.read(), b'%f\n' % 1.0)

    @pytest.mark.skipif(sys.platform == 'win32', reason='files>4GB may not work')
    @pytest.mark.slow
    @requires_memory(free_bytes=7000000000.0)
    def test_large_zip(self):

        def check_large_zip(memoryerror_raised):
            memoryerror_raised.value = False
            try:
                test_data = np.asarray([np.random.rand(np.random.randint(50, 100), 4) for i in range(800000)], dtype=object)
                with tempdir() as tmpdir:
                    np.savez(os.path.join(tmpdir, 'test.npz'), test_data=test_data)
            except MemoryError:
                memoryerror_raised.value = True
                raise
        memoryerror_raised = Value(c_bool)
        ctx = get_context('fork')
        p = ctx.Process(target=check_large_zip, args=(memoryerror_raised,))
        p.start()
        p.join()
        if memoryerror_raised.value:
            raise MemoryError('Child process raised a MemoryError exception')
        if p.exitcode == -9:
            pytest.xfail('subprocess got a SIGKILL, apparently free memory was not sufficient')
        assert p.exitcode == 0