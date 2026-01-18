from __future__ import annotations
import collections.abc
import tempfile
import sys
import warnings
import operator
import io
import itertools
import functools
import ctypes
import os
import gc
import re
import weakref
import pytest
from contextlib import contextmanager
from numpy.compat import pickle
import pathlib
import builtins
from decimal import Decimal
import mmap
import numpy as np
import numpy.core._multiarray_tests as _multiarray_tests
from numpy.core._rational_tests import rational
from numpy.testing import (
from numpy.testing._private.utils import requires_memory, _no_tracing
from numpy.core.tests._locales import CommaDecimalPointLocale
from numpy.lib.recfunctions import repack_fields
from numpy.core.multiarray import _get_ndarray_c_version
from datetime import timedelta, datetime
from numpy.core._internal import _dtype_from_pep3118
from numpy.testing import IS_PYPY
class TestIO:
    """Test tofile, fromfile, tobytes, and fromstring"""

    @pytest.fixture()
    def x(self):
        shape = (2, 4, 3)
        rand = np.random.random
        x = rand(shape) + rand(shape).astype(complex) * 1j
        x[0, :, 1] = [np.nan, np.inf, -np.inf, np.nan]
        return x

    @pytest.fixture(params=['string', 'path_obj'])
    def tmp_filename(self, tmp_path, request):
        filename = tmp_path / 'file'
        if request.param == 'string':
            filename = str(filename)
        yield filename

    def test_nofile(self):
        b = io.BytesIO()
        assert_raises(OSError, np.fromfile, b, np.uint8, 80)
        d = np.ones(7)
        assert_raises(OSError, lambda x: x.tofile(b), d)

    def test_bool_fromstring(self):
        v = np.array([True, False, True, False], dtype=np.bool_)
        y = np.fromstring('1 0 -2.3 0.0', sep=' ', dtype=np.bool_)
        assert_array_equal(v, y)

    def test_uint64_fromstring(self):
        d = np.fromstring('9923372036854775807 104783749223640', dtype=np.uint64, sep=' ')
        e = np.array([9923372036854775807, 104783749223640], dtype=np.uint64)
        assert_array_equal(d, e)

    def test_int64_fromstring(self):
        d = np.fromstring('-25041670086757 104783749223640', dtype=np.int64, sep=' ')
        e = np.array([-25041670086757, 104783749223640], dtype=np.int64)
        assert_array_equal(d, e)

    def test_fromstring_count0(self):
        d = np.fromstring('1,2', sep=',', dtype=np.int64, count=0)
        assert d.shape == (0,)

    def test_empty_files_text(self, tmp_filename):
        with open(tmp_filename, 'w') as f:
            pass
        y = np.fromfile(tmp_filename)
        assert_(y.size == 0, 'Array not empty')

    def test_empty_files_binary(self, tmp_filename):
        with open(tmp_filename, 'wb') as f:
            pass
        y = np.fromfile(tmp_filename, sep=' ')
        assert_(y.size == 0, 'Array not empty')

    def test_roundtrip_file(self, x, tmp_filename):
        with open(tmp_filename, 'wb') as f:
            x.tofile(f)
        with open(tmp_filename, 'rb') as f:
            y = np.fromfile(f, dtype=x.dtype)
        assert_array_equal(y, x.flat)

    def test_roundtrip(self, x, tmp_filename):
        x.tofile(tmp_filename)
        y = np.fromfile(tmp_filename, dtype=x.dtype)
        assert_array_equal(y, x.flat)

    def test_roundtrip_dump_pathlib(self, x, tmp_filename):
        p = pathlib.Path(tmp_filename)
        x.dump(p)
        y = np.load(p, allow_pickle=True)
        assert_array_equal(y, x)

    def test_roundtrip_binary_str(self, x):
        s = x.tobytes()
        y = np.frombuffer(s, dtype=x.dtype)
        assert_array_equal(y, x.flat)
        s = x.tobytes('F')
        y = np.frombuffer(s, dtype=x.dtype)
        assert_array_equal(y, x.flatten('F'))

    def test_roundtrip_str(self, x):
        x = x.real.ravel()
        s = '@'.join(map(str, x))
        y = np.fromstring(s, sep='@')
        nan_mask = ~np.isfinite(x)
        assert_array_equal(x[nan_mask], y[nan_mask])
        assert_array_almost_equal(x[~nan_mask], y[~nan_mask], decimal=5)

    def test_roundtrip_repr(self, x):
        x = x.real.ravel()
        s = '@'.join(map(repr, x))
        y = np.fromstring(s, sep='@')
        assert_array_equal(x, y)

    def test_unseekable_fromfile(self, x, tmp_filename):
        x.tofile(tmp_filename)

        def fail(*args, **kwargs):
            raise OSError('Can not tell or seek')
        with io.open(tmp_filename, 'rb', buffering=0) as f:
            f.seek = fail
            f.tell = fail
            assert_raises(OSError, np.fromfile, f, dtype=x.dtype)

    def test_io_open_unbuffered_fromfile(self, x, tmp_filename):
        x.tofile(tmp_filename)
        with io.open(tmp_filename, 'rb', buffering=0) as f:
            y = np.fromfile(f, dtype=x.dtype)
            assert_array_equal(y, x.flat)

    def test_largish_file(self, tmp_filename):
        d = np.zeros(4 * 1024 ** 2)
        d.tofile(tmp_filename)
        assert_equal(os.path.getsize(tmp_filename), d.nbytes)
        assert_array_equal(d, np.fromfile(tmp_filename))
        with open(tmp_filename, 'r+b') as f:
            f.seek(d.nbytes)
            d.tofile(f)
            assert_equal(os.path.getsize(tmp_filename), d.nbytes * 2)
        open(tmp_filename, 'w').close()
        with open(tmp_filename, 'ab') as f:
            d.tofile(f)
        assert_array_equal(d, np.fromfile(tmp_filename))
        with open(tmp_filename, 'ab') as f:
            d.tofile(f)
        assert_equal(os.path.getsize(tmp_filename), d.nbytes * 2)

    def test_io_open_buffered_fromfile(self, x, tmp_filename):
        x.tofile(tmp_filename)
        with io.open(tmp_filename, 'rb', buffering=-1) as f:
            y = np.fromfile(f, dtype=x.dtype)
        assert_array_equal(y, x.flat)

    def test_file_position_after_fromfile(self, tmp_filename):
        sizes = [io.DEFAULT_BUFFER_SIZE // 8, io.DEFAULT_BUFFER_SIZE, io.DEFAULT_BUFFER_SIZE * 8]
        for size in sizes:
            with open(tmp_filename, 'wb') as f:
                f.seek(size - 1)
                f.write(b'\x00')
            for mode in ['rb', 'r+b']:
                err_msg = '%d %s' % (size, mode)
                with open(tmp_filename, mode) as f:
                    f.read(2)
                    np.fromfile(f, dtype=np.float64, count=1)
                    pos = f.tell()
                assert_equal(pos, 10, err_msg=err_msg)

    def test_file_position_after_tofile(self, tmp_filename):
        sizes = [io.DEFAULT_BUFFER_SIZE // 8, io.DEFAULT_BUFFER_SIZE, io.DEFAULT_BUFFER_SIZE * 8]
        for size in sizes:
            err_msg = '%d' % (size,)
            with open(tmp_filename, 'wb') as f:
                f.seek(size - 1)
                f.write(b'\x00')
                f.seek(10)
                f.write(b'12')
                np.array([0], dtype=np.float64).tofile(f)
                pos = f.tell()
            assert_equal(pos, 10 + 2 + 8, err_msg=err_msg)
            with open(tmp_filename, 'r+b') as f:
                f.read(2)
                f.seek(0, 1)
                np.array([0], dtype=np.float64).tofile(f)
                pos = f.tell()
            assert_equal(pos, 10, err_msg=err_msg)

    def test_load_object_array_fromfile(self, tmp_filename):
        with open(tmp_filename, 'w') as f:
            pass
        with open(tmp_filename, 'rb') as f:
            assert_raises_regex(ValueError, 'Cannot read into object array', np.fromfile, f, dtype=object)
        assert_raises_regex(ValueError, 'Cannot read into object array', np.fromfile, tmp_filename, dtype=object)

    def test_fromfile_offset(self, x, tmp_filename):
        with open(tmp_filename, 'wb') as f:
            x.tofile(f)
        with open(tmp_filename, 'rb') as f:
            y = np.fromfile(f, dtype=x.dtype, offset=0)
            assert_array_equal(y, x.flat)
        with open(tmp_filename, 'rb') as f:
            count_items = len(x.flat) // 8
            offset_items = len(x.flat) // 4
            offset_bytes = x.dtype.itemsize * offset_items
            y = np.fromfile(f, dtype=x.dtype, count=count_items, offset=offset_bytes)
            assert_array_equal(y, x.flat[offset_items:offset_items + count_items])
            offset_bytes = x.dtype.itemsize
            z = np.fromfile(f, dtype=x.dtype, offset=offset_bytes)
            assert_array_equal(z, x.flat[offset_items + count_items + 1:])
        with open(tmp_filename, 'wb') as f:
            x.tofile(f, sep=',')
        with open(tmp_filename, 'rb') as f:
            assert_raises_regex(TypeError, "'offset' argument only permitted for binary files", np.fromfile, tmp_filename, dtype=x.dtype, sep=',', offset=1)

    @pytest.mark.skipif(IS_PYPY, reason="bug in PyPy's PyNumber_AsSsize_t")
    def test_fromfile_bad_dup(self, x, tmp_filename):

        def dup_str(fd):
            return 'abc'

        def dup_bigint(fd):
            return 2 ** 68
        old_dup = os.dup
        try:
            with open(tmp_filename, 'wb') as f:
                x.tofile(f)
                for dup, exc in ((dup_str, TypeError), (dup_bigint, OSError)):
                    os.dup = dup
                    assert_raises(exc, np.fromfile, f)
        finally:
            os.dup = old_dup

    def _check_from(self, s, value, filename, **kw):
        if 'sep' not in kw:
            y = np.frombuffer(s, **kw)
        else:
            y = np.fromstring(s, **kw)
        assert_array_equal(y, value)
        with open(filename, 'wb') as f:
            f.write(s)
        y = np.fromfile(filename, **kw)
        assert_array_equal(y, value)

    @pytest.fixture(params=['period', 'comma'])
    def decimal_sep_localization(self, request):
        """
        Including this fixture in a test will automatically
        execute it with both types of decimal separator.

        So::

            def test_decimal(decimal_sep_localization):
                pass

        is equivalent to the following two tests::

            def test_decimal_period_separator():
                pass

            def test_decimal_comma_separator():
                with CommaDecimalPointLocale():
                    pass
        """
        if request.param == 'period':
            yield
        elif request.param == 'comma':
            with CommaDecimalPointLocale():
                yield
        else:
            assert False, request.param

    def test_nan(self, tmp_filename, decimal_sep_localization):
        self._check_from(b'nan +nan -nan NaN nan(foo) +NaN(BAR) -NAN(q_u_u_x_)', [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan], tmp_filename, sep=' ')

    def test_inf(self, tmp_filename, decimal_sep_localization):
        self._check_from(b'inf +inf -inf infinity -Infinity iNfInItY -inF', [np.inf, np.inf, -np.inf, np.inf, -np.inf, np.inf, -np.inf], tmp_filename, sep=' ')

    def test_numbers(self, tmp_filename, decimal_sep_localization):
        self._check_from(b'1.234 -1.234 .3 .3e55 -123133.1231e+133', [1.234, -1.234, 0.3, 3e+54, -1.231331231e+138], tmp_filename, sep=' ')

    def test_binary(self, tmp_filename):
        self._check_from(b'\x00\x00\x80?\x00\x00\x00@\x00\x00@@\x00\x00\x80@', np.array([1, 2, 3, 4]), tmp_filename, dtype='<f4')

    def test_string(self, tmp_filename):
        self._check_from(b'1,2,3,4', [1.0, 2.0, 3.0, 4.0], tmp_filename, sep=',')

    def test_counted_string(self, tmp_filename, decimal_sep_localization):
        self._check_from(b'1,2,3,4', [1.0, 2.0, 3.0, 4.0], tmp_filename, count=4, sep=',')
        self._check_from(b'1,2,3,4', [1.0, 2.0, 3.0], tmp_filename, count=3, sep=',')
        self._check_from(b'1,2,3,4', [1.0, 2.0, 3.0, 4.0], tmp_filename, count=-1, sep=',')

    def test_string_with_ws(self, tmp_filename):
        self._check_from(b'1 2  3     4   ', [1, 2, 3, 4], tmp_filename, dtype=int, sep=' ')

    def test_counted_string_with_ws(self, tmp_filename):
        self._check_from(b'1 2  3     4   ', [1, 2, 3], tmp_filename, count=3, dtype=int, sep=' ')

    def test_ascii(self, tmp_filename, decimal_sep_localization):
        self._check_from(b'1 , 2 , 3 , 4', [1.0, 2.0, 3.0, 4.0], tmp_filename, sep=',')
        self._check_from(b'1,2,3,4', [1.0, 2.0, 3.0, 4.0], tmp_filename, dtype=float, sep=',')

    def test_malformed(self, tmp_filename, decimal_sep_localization):
        with assert_warns(DeprecationWarning):
            self._check_from(b'1.234 1,234', [1.234, 1.0], tmp_filename, sep=' ')

    def test_long_sep(self, tmp_filename):
        self._check_from(b'1_x_3_x_4_x_5', [1, 3, 4, 5], tmp_filename, sep='_x_')

    def test_dtype(self, tmp_filename):
        v = np.array([1, 2, 3, 4], dtype=np.int_)
        self._check_from(b'1,2,3,4', v, tmp_filename, sep=',', dtype=np.int_)

    def test_dtype_bool(self, tmp_filename):
        v = np.array([True, False, True, False], dtype=np.bool_)
        s = b'1,0,-2.3,0'
        with open(tmp_filename, 'wb') as f:
            f.write(s)
        y = np.fromfile(tmp_filename, sep=',', dtype=np.bool_)
        assert_(y.dtype == '?')
        assert_array_equal(y, v)

    def test_tofile_sep(self, tmp_filename, decimal_sep_localization):
        x = np.array([1.51, 2, 3.51, 4], dtype=float)
        with open(tmp_filename, 'w') as f:
            x.tofile(f, sep=',')
        with open(tmp_filename, 'r') as f:
            s = f.read()
        y = np.array([float(p) for p in s.split(',')])
        assert_array_equal(x, y)

    def test_tofile_format(self, tmp_filename, decimal_sep_localization):
        x = np.array([1.51, 2, 3.51, 4], dtype=float)
        with open(tmp_filename, 'w') as f:
            x.tofile(f, sep=',', format='%.2f')
        with open(tmp_filename, 'r') as f:
            s = f.read()
        assert_equal(s, '1.51,2.00,3.51,4.00')

    def test_tofile_cleanup(self, tmp_filename):
        x = np.zeros(10, dtype=object)
        with open(tmp_filename, 'wb') as f:
            assert_raises(OSError, lambda: x.tofile(f, sep=''))
        os.remove(tmp_filename)
        assert_raises(OSError, lambda: x.tofile(tmp_filename))
        os.remove(tmp_filename)

    def test_fromfile_subarray_binary(self, tmp_filename):
        x = np.arange(24, dtype='i4').reshape(2, 3, 4)
        x.tofile(tmp_filename)
        res = np.fromfile(tmp_filename, dtype='(3,4)i4')
        assert_array_equal(x, res)
        x_str = x.tobytes()
        with assert_warns(DeprecationWarning):
            res = np.fromstring(x_str, dtype='(3,4)i4')
            assert_array_equal(x, res)

    def test_parsing_subarray_unsupported(self, tmp_filename):
        data = '12,42,13,' * 50
        with pytest.raises(ValueError):
            expected = np.fromstring(data, dtype='(3,)i', sep=',')
        with open(tmp_filename, 'w') as f:
            f.write(data)
        with pytest.raises(ValueError):
            np.fromfile(tmp_filename, dtype='(3,)i', sep=',')

    def test_read_shorter_than_count_subarray(self, tmp_filename):
        expected = np.arange(511 * 10, dtype='i').reshape(-1, 10)
        binary = expected.tobytes()
        with pytest.raises(ValueError):
            with pytest.warns(DeprecationWarning):
                np.fromstring(binary, dtype='(10,)i', count=10000)
        expected.tofile(tmp_filename)
        res = np.fromfile(tmp_filename, dtype='(10,)i', count=10000)
        assert_array_equal(res, expected)