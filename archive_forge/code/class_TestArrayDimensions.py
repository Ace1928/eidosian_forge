from os import path
import warnings
import numpy as np
from numpy.testing import (assert_equal, assert_array_equal,
import pytest
from scipy.io import readsav
from scipy.io import _idl
class TestArrayDimensions:

    def test_1d(self):
        s = readsav(path.join(DATA_PATH, 'array_float32_1d.sav'), verbose=False)
        assert_equal(s.array1d.shape, (123,))

    def test_2d(self):
        s = readsav(path.join(DATA_PATH, 'array_float32_2d.sav'), verbose=False)
        assert_equal(s.array2d.shape, (22, 12))

    def test_3d(self):
        s = readsav(path.join(DATA_PATH, 'array_float32_3d.sav'), verbose=False)
        assert_equal(s.array3d.shape, (11, 22, 12))

    def test_4d(self):
        s = readsav(path.join(DATA_PATH, 'array_float32_4d.sav'), verbose=False)
        assert_equal(s.array4d.shape, (4, 5, 8, 7))

    def test_5d(self):
        s = readsav(path.join(DATA_PATH, 'array_float32_5d.sav'), verbose=False)
        assert_equal(s.array5d.shape, (4, 3, 4, 6, 5))

    def test_6d(self):
        s = readsav(path.join(DATA_PATH, 'array_float32_6d.sav'), verbose=False)
        assert_equal(s.array6d.shape, (3, 6, 4, 5, 3, 4))

    def test_7d(self):
        s = readsav(path.join(DATA_PATH, 'array_float32_7d.sav'), verbose=False)
        assert_equal(s.array7d.shape, (2, 1, 2, 3, 4, 3, 2))

    def test_8d(self):
        s = readsav(path.join(DATA_PATH, 'array_float32_8d.sav'), verbose=False)
        assert_equal(s.array8d.shape, (4, 3, 2, 1, 2, 3, 5, 4))