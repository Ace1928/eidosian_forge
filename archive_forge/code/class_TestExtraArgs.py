import pytest
import numpy as np
from numpy.testing import (
class TestExtraArgs:

    def test_superclass(self):
        s = np.str_(b'\\x61', encoding='unicode-escape')
        assert s == 'a'
        s = np.str_(b'\\x61', 'unicode-escape')
        assert s == 'a'
        with pytest.raises(UnicodeDecodeError):
            np.str_(b'\\xx', encoding='unicode-escape')
        with pytest.raises(UnicodeDecodeError):
            np.str_(b'\\xx', 'unicode-escape')
        assert np.bytes_(-2) == b'-2'

    def test_datetime(self):
        dt = np.datetime64('2000-01', ('M', 2))
        assert np.datetime_data(dt) == ('M', 2)
        with pytest.raises(TypeError):
            np.datetime64('2000', garbage=True)

    def test_bool(self):
        with pytest.raises(TypeError):
            np.bool_(False, garbage=True)

    def test_void(self):
        with pytest.raises(TypeError):
            np.void(b'test', garbage=True)