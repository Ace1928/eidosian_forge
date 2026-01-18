import sys
import itertools
import pytest
import numpy as np
from numpy.testing import assert_, assert_equal, assert_raises, IS_PYPY
class CreateZeros:
    """Check the creation of heterogeneous arrays zero-valued"""

    def test_zeros0D(self):
        """Check creation of 0-dimensional objects"""
        h = np.zeros((), dtype=self._descr)
        assert_(normalize_descr(self._descr) == h.dtype.descr)
        assert_(h.dtype.fields['x'][0].name[:4] == 'void')
        assert_(h.dtype.fields['x'][0].char == 'V')
        assert_(h.dtype.fields['x'][0].type == np.void)
        assert_equal(h['z'], np.zeros((), dtype='u1'))

    def test_zerosSD(self):
        """Check creation of single-dimensional objects"""
        h = np.zeros((2,), dtype=self._descr)
        assert_(normalize_descr(self._descr) == h.dtype.descr)
        assert_(h.dtype['y'].name[:4] == 'void')
        assert_(h.dtype['y'].char == 'V')
        assert_(h.dtype['y'].type == np.void)
        assert_equal(h['z'], np.zeros((2,), dtype='u1'))

    def test_zerosMD(self):
        """Check creation of multi-dimensional objects"""
        h = np.zeros((2, 3), dtype=self._descr)
        assert_(normalize_descr(self._descr) == h.dtype.descr)
        assert_(h.dtype['z'].name == 'uint8')
        assert_(h.dtype['z'].char == 'B')
        assert_(h.dtype['z'].type == np.uint8)
        assert_equal(h['z'], np.zeros((2, 3), dtype='u1'))