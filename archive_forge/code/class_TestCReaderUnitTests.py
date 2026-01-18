import sys
import os
import pytest
from tempfile import NamedTemporaryFile, mkstemp
from io import StringIO
import numpy as np
from numpy.ma.testutils import assert_equal
from numpy.testing import assert_array_equal, HAS_REFCOUNT, IS_PYPY
class TestCReaderUnitTests:

    def test_not_an_filelike(self):
        with pytest.raises(AttributeError, match='.*read'):
            np.core._multiarray_umath._load_from_filelike(object(), dtype=np.dtype('i'), filelike=True)

    def test_filelike_read_fails(self):

        class BadFileLike:
            counter = 0

            def read(self, size):
                self.counter += 1
                if self.counter > 20:
                    raise RuntimeError('Bad bad bad!')
                return '1,2,3\n'
        with pytest.raises(RuntimeError, match='Bad bad bad!'):
            np.core._multiarray_umath._load_from_filelike(BadFileLike(), dtype=np.dtype('i'), filelike=True)

    def test_filelike_bad_read(self):

        class BadFileLike:
            counter = 0

            def read(self, size):
                return 1234
        with pytest.raises(TypeError, match='non-string returned while reading data'):
            np.core._multiarray_umath._load_from_filelike(BadFileLike(), dtype=np.dtype('i'), filelike=True)

    def test_not_an_iter(self):
        with pytest.raises(TypeError, match='error reading from object, expected an iterable'):
            np.core._multiarray_umath._load_from_filelike(object(), dtype=np.dtype('i'), filelike=False)

    def test_bad_type(self):
        with pytest.raises(TypeError, match='internal error: dtype must'):
            np.core._multiarray_umath._load_from_filelike(object(), dtype='i', filelike=False)

    def test_bad_encoding(self):
        with pytest.raises(TypeError, match='encoding must be a unicode'):
            np.core._multiarray_umath._load_from_filelike(object(), dtype=np.dtype('i'), filelike=False, encoding=123)

    @pytest.mark.parametrize('newline', ['\r', '\n', '\r\n'])
    def test_manual_universal_newlines(self, newline):
        data = StringIO('0\n1\n"2\n"\n3\n4 #\n'.replace('\n', newline), newline='')
        res = np.core._multiarray_umath._load_from_filelike(data, dtype=np.dtype('U10'), filelike=True, quote='"', comment='#', skiplines=1)
        assert_array_equal(res[:, 0], ['1', f'2{newline}', '3', '4 '])