import pytest
import numpy as np
import numpy.ma as ma
from numpy.ma.mrecords import MaskedRecords
from numpy.ma.testutils import assert_equal
from numpy.testing import assert_, assert_raises
from numpy.lib.recfunctions import (
class TestJoinBy2:

    @classmethod
    def setup_method(cls):
        cls.a = np.array(list(zip(np.arange(10), np.arange(50, 60), np.arange(100, 110))), dtype=[('a', int), ('b', int), ('c', int)])
        cls.b = np.array(list(zip(np.arange(10), np.arange(65, 75), np.arange(100, 110))), dtype=[('a', int), ('b', int), ('d', int)])

    def test_no_r1postfix(self):
        a, b = (self.a, self.b)
        test = join_by('a', a, b, r1postfix='', r2postfix='2', jointype='inner')
        control = np.array([(0, 50, 65, 100, 100), (1, 51, 66, 101, 101), (2, 52, 67, 102, 102), (3, 53, 68, 103, 103), (4, 54, 69, 104, 104), (5, 55, 70, 105, 105), (6, 56, 71, 106, 106), (7, 57, 72, 107, 107), (8, 58, 73, 108, 108), (9, 59, 74, 109, 109)], dtype=[('a', int), ('b', int), ('b2', int), ('c', int), ('d', int)])
        assert_equal(test, control)

    def test_no_postfix(self):
        assert_raises(ValueError, join_by, 'a', self.a, self.b, r1postfix='', r2postfix='')

    def test_no_r2postfix(self):
        a, b = (self.a, self.b)
        test = join_by('a', a, b, r1postfix='1', r2postfix='', jointype='inner')
        control = np.array([(0, 50, 65, 100, 100), (1, 51, 66, 101, 101), (2, 52, 67, 102, 102), (3, 53, 68, 103, 103), (4, 54, 69, 104, 104), (5, 55, 70, 105, 105), (6, 56, 71, 106, 106), (7, 57, 72, 107, 107), (8, 58, 73, 108, 108), (9, 59, 74, 109, 109)], dtype=[('a', int), ('b1', int), ('b', int), ('c', int), ('d', int)])
        assert_equal(test, control)

    def test_two_keys_two_vars(self):
        a = np.array(list(zip(np.tile([10, 11], 5), np.repeat(np.arange(5), 2), np.arange(50, 60), np.arange(10, 20))), dtype=[('k', int), ('a', int), ('b', int), ('c', int)])
        b = np.array(list(zip(np.tile([10, 11], 5), np.repeat(np.arange(5), 2), np.arange(65, 75), np.arange(0, 10))), dtype=[('k', int), ('a', int), ('b', int), ('c', int)])
        control = np.array([(10, 0, 50, 65, 10, 0), (11, 0, 51, 66, 11, 1), (10, 1, 52, 67, 12, 2), (11, 1, 53, 68, 13, 3), (10, 2, 54, 69, 14, 4), (11, 2, 55, 70, 15, 5), (10, 3, 56, 71, 16, 6), (11, 3, 57, 72, 17, 7), (10, 4, 58, 73, 18, 8), (11, 4, 59, 74, 19, 9)], dtype=[('k', int), ('a', int), ('b1', int), ('b2', int), ('c1', int), ('c2', int)])
        test = join_by(['a', 'k'], a, b, r1postfix='1', r2postfix='2', jointype='inner')
        assert_equal(test.dtype, control.dtype)
        assert_equal(test, control)