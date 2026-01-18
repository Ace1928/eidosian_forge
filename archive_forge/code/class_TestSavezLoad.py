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
class TestSavezLoad(RoundtripTest):

    def roundtrip(self, *args, **kwargs):
        RoundtripTest.roundtrip(self, np.savez, *args, **kwargs)
        try:
            for n, arr in enumerate(self.arr):
                reloaded = self.arr_reloaded['arr_%d' % n]
                assert_equal(arr, reloaded)
                assert_equal(arr.dtype, reloaded.dtype)
                assert_equal(arr.flags.fnc, reloaded.flags.fnc)
        finally:
            if self.arr_reloaded.fid:
                self.arr_reloaded.fid.close()
                os.remove(self.arr_reloaded.fid.name)

    @pytest.mark.skipif(IS_PYPY, reason='Hangs on PyPy')
    @pytest.mark.skipif(not IS_64BIT, reason='Needs 64bit platform')
    @pytest.mark.slow
    def test_big_arrays(self):
        L = (1 << 31) + 100000
        a = np.empty(L, dtype=np.uint8)
        with temppath(prefix='numpy_test_big_arrays_', suffix='.npz') as tmp:
            np.savez(tmp, a=a)
            del a
            npfile = np.load(tmp)
            a = npfile['a']
            npfile.close()
            del a

    def test_multiple_arrays(self):
        a = np.array([[1, 2], [3, 4]], float)
        b = np.array([[1 + 2j, 2 + 7j], [3 - 6j, 4 + 12j]], complex)
        self.roundtrip(a, b)

    def test_named_arrays(self):
        a = np.array([[1, 2], [3, 4]], float)
        b = np.array([[1 + 2j, 2 + 7j], [3 - 6j, 4 + 12j]], complex)
        c = BytesIO()
        np.savez(c, file_a=a, file_b=b)
        c.seek(0)
        l = np.load(c)
        assert_equal(a, l['file_a'])
        assert_equal(b, l['file_b'])

    def test_tuple_getitem_raises(self):
        a = np.array([1, 2, 3])
        f = BytesIO()
        np.savez(f, a=a)
        f.seek(0)
        l = np.load(f)
        with pytest.raises(KeyError, match='(1, 2)'):
            l[1, 2]

    def test_BagObj(self):
        a = np.array([[1, 2], [3, 4]], float)
        b = np.array([[1 + 2j, 2 + 7j], [3 - 6j, 4 + 12j]], complex)
        c = BytesIO()
        np.savez(c, file_a=a, file_b=b)
        c.seek(0)
        l = np.load(c)
        assert_equal(sorted(dir(l.f)), ['file_a', 'file_b'])
        assert_equal(a, l.f.file_a)
        assert_equal(b, l.f.file_b)

    @pytest.mark.skipif(IS_WASM, reason='Cannot start thread')
    def test_savez_filename_clashes(self):

        def writer(error_list):
            with temppath(suffix='.npz') as tmp:
                arr = np.random.randn(500, 500)
                try:
                    np.savez(tmp, arr=arr)
                except OSError as err:
                    error_list.append(err)
        errors = []
        threads = [threading.Thread(target=writer, args=(errors,)) for j in range(3)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        if errors:
            raise AssertionError(errors)

    def test_not_closing_opened_fid(self):
        with temppath(suffix='.npz') as tmp:
            with open(tmp, 'wb') as fp:
                np.savez(fp, data='LOVELY LOAD')
            with open(tmp, 'rb', 10000) as fp:
                fp.seek(0)
                assert_(not fp.closed)
                np.load(fp)['data']
                assert_(not fp.closed)
                fp.seek(0)
                assert_(not fp.closed)

    @pytest.mark.slow_pypy
    def test_closing_fid(self):
        with temppath(suffix='.npz') as tmp:
            np.savez(tmp, data='LOVELY LOAD')
            with suppress_warnings() as sup:
                sup.filter(ResourceWarning)
                for i in range(1, 1025):
                    try:
                        np.load(tmp)['data']
                    except Exception as e:
                        msg = 'Failed to load data from a file: %s' % e
                        raise AssertionError(msg)
                    finally:
                        if IS_PYPY:
                            gc.collect()

    def test_closing_zipfile_after_load(self):
        prefix = 'numpy_test_closing_zipfile_after_load_'
        with temppath(suffix='.npz', prefix=prefix) as tmp:
            np.savez(tmp, lab='place holder')
            data = np.load(tmp)
            fp = data.zip.fp
            data.close()
            assert_(fp.closed)

    @pytest.mark.parametrize('count, expected_repr', [(1, 'NpzFile {fname!r} with keys: arr_0'), (5, 'NpzFile {fname!r} with keys: arr_0, arr_1, arr_2, arr_3, arr_4'), (6, 'NpzFile {fname!r} with keys: arr_0, arr_1, arr_2, arr_3, arr_4...')])
    def test_repr_lists_keys(self, count, expected_repr):
        a = np.array([[1, 2], [3, 4]], float)
        with temppath(suffix='.npz') as tmp:
            np.savez(tmp, *[a] * count)
            l = np.load(tmp)
            assert repr(l) == expected_repr.format(fname=tmp)
            l.close()