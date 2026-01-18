import sys
import itertools
import pytest
import numpy as np
from numpy.testing import assert_, assert_equal, assert_raises, IS_PYPY
class ReadValuesNested:
    """Check the reading of values in heterogeneous arrays (nested)"""

    def test_access_top_fields(self):
        """Check reading the top fields of a nested array"""
        h = np.array(self._buffer, dtype=self._descr)
        if not self.multiple_rows:
            assert_(h.shape == ())
            assert_equal(h['x'], np.array(self._buffer[0], dtype='i4'))
            assert_equal(h['y'], np.array(self._buffer[4], dtype='f8'))
            assert_equal(h['z'], np.array(self._buffer[5], dtype='u1'))
        else:
            assert_(len(h) == 2)
            assert_equal(h['x'], np.array([self._buffer[0][0], self._buffer[1][0]], dtype='i4'))
            assert_equal(h['y'], np.array([self._buffer[0][4], self._buffer[1][4]], dtype='f8'))
            assert_equal(h['z'], np.array([self._buffer[0][5], self._buffer[1][5]], dtype='u1'))

    def test_nested1_acessors(self):
        """Check reading the nested fields of a nested array (1st level)"""
        h = np.array(self._buffer, dtype=self._descr)
        if not self.multiple_rows:
            assert_equal(h['Info']['value'], np.array(self._buffer[1][0], dtype='c16'))
            assert_equal(h['Info']['y2'], np.array(self._buffer[1][1], dtype='f8'))
            assert_equal(h['info']['Name'], np.array(self._buffer[3][0], dtype='U2'))
            assert_equal(h['info']['Value'], np.array(self._buffer[3][1], dtype='c16'))
        else:
            assert_equal(h['Info']['value'], np.array([self._buffer[0][1][0], self._buffer[1][1][0]], dtype='c16'))
            assert_equal(h['Info']['y2'], np.array([self._buffer[0][1][1], self._buffer[1][1][1]], dtype='f8'))
            assert_equal(h['info']['Name'], np.array([self._buffer[0][3][0], self._buffer[1][3][0]], dtype='U2'))
            assert_equal(h['info']['Value'], np.array([self._buffer[0][3][1], self._buffer[1][3][1]], dtype='c16'))

    def test_nested2_acessors(self):
        """Check reading the nested fields of a nested array (2nd level)"""
        h = np.array(self._buffer, dtype=self._descr)
        if not self.multiple_rows:
            assert_equal(h['Info']['Info2']['value'], np.array(self._buffer[1][2][1], dtype='c16'))
            assert_equal(h['Info']['Info2']['z3'], np.array(self._buffer[1][2][3], dtype='u4'))
        else:
            assert_equal(h['Info']['Info2']['value'], np.array([self._buffer[0][1][2][1], self._buffer[1][1][2][1]], dtype='c16'))
            assert_equal(h['Info']['Info2']['z3'], np.array([self._buffer[0][1][2][3], self._buffer[1][1][2][3]], dtype='u4'))

    def test_nested1_descriptor(self):
        """Check access nested descriptors of a nested array (1st level)"""
        h = np.array(self._buffer, dtype=self._descr)
        assert_(h.dtype['Info']['value'].name == 'complex128')
        assert_(h.dtype['Info']['y2'].name == 'float64')
        assert_(h.dtype['info']['Name'].name == 'str256')
        assert_(h.dtype['info']['Value'].name == 'complex128')

    def test_nested2_descriptor(self):
        """Check access nested descriptors of a nested array (2nd level)"""
        h = np.array(self._buffer, dtype=self._descr)
        assert_(h.dtype['Info']['Info2']['value'].name == 'void256')
        assert_(h.dtype['Info']['Info2']['z3'].name == 'void64')