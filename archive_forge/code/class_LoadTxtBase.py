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
class LoadTxtBase:

    def check_compressed(self, fopen, suffixes):
        wanted = np.arange(6).reshape((2, 3))
        linesep = ('\n', '\r\n', '\r')
        for sep in linesep:
            data = '0 1 2' + sep + '3 4 5'
            for suffix in suffixes:
                with temppath(suffix=suffix) as name:
                    with fopen(name, mode='wt', encoding='UTF-32-LE') as f:
                        f.write(data)
                    res = self.loadfunc(name, encoding='UTF-32-LE')
                    assert_array_equal(res, wanted)
                    with fopen(name, 'rt', encoding='UTF-32-LE') as f:
                        res = self.loadfunc(f)
                    assert_array_equal(res, wanted)

    def test_compressed_gzip(self):
        self.check_compressed(gzip.open, ('.gz',))

    @pytest.mark.skipif(not HAS_BZ2, reason='Needs bz2')
    def test_compressed_bz2(self):
        self.check_compressed(bz2.open, ('.bz2',))

    @pytest.mark.skipif(not HAS_LZMA, reason='Needs lzma')
    def test_compressed_lzma(self):
        self.check_compressed(lzma.open, ('.xz', '.lzma'))

    def test_encoding(self):
        with temppath() as path:
            with open(path, 'wb') as f:
                f.write('0.\n1.\n2.'.encode('UTF-16'))
            x = self.loadfunc(path, encoding='UTF-16')
            assert_array_equal(x, [0.0, 1.0, 2.0])

    def test_stringload(self):
        nonascii = b'\xc3\xb6\xc3\xbc\xc3\xb6'.decode('UTF-8')
        with temppath() as path:
            with open(path, 'wb') as f:
                f.write(nonascii.encode('UTF-16'))
            x = self.loadfunc(path, encoding='UTF-16', dtype=np.str_)
            assert_array_equal(x, nonascii)

    def test_binary_decode(self):
        utf16 = b'\xff\xfeh\x04 \x00i\x04 \x00j\x04'
        v = self.loadfunc(BytesIO(utf16), dtype=np.str_, encoding='UTF-16')
        assert_array_equal(v, np.array(utf16.decode('UTF-16').split()))

    def test_converters_decode(self):
        c = TextIO()
        c.write(b'\xcf\x96')
        c.seek(0)
        x = self.loadfunc(c, dtype=np.str_, converters={0: lambda x: x.decode('UTF-8')})
        a = np.array([b'\xcf\x96'.decode('UTF-8')])
        assert_array_equal(x, a)

    def test_converters_nodecode(self):
        utf8 = b'\xcf\x96'.decode('UTF-8')
        with temppath() as path:
            with io.open(path, 'wt', encoding='UTF-8') as f:
                f.write(utf8)
            x = self.loadfunc(path, dtype=np.str_, converters={0: lambda x: x + 't'}, encoding='UTF-8')
            a = np.array([utf8 + 't'])
            assert_array_equal(x, a)