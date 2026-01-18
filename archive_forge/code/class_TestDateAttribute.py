import datetime
import os
import sys
from os.path import join as pjoin
from io import StringIO
import numpy as np
from numpy.testing import (assert_array_almost_equal,
from pytest import raises as assert_raises
from scipy.io.arff import loadarff
from scipy.io.arff._arffread import read_header, ParseArffError
class TestDateAttribute:

    def setup_method(self):
        self.data, self.meta = loadarff(test7)

    def test_year_attribute(self):
        expected = np.array(['1999', '2004', '1817', '2100', '2013', '1631'], dtype='datetime64[Y]')
        assert_array_equal(self.data['attr_year'], expected)

    def test_month_attribute(self):
        expected = np.array(['1999-01', '2004-12', '1817-04', '2100-09', '2013-11', '1631-10'], dtype='datetime64[M]')
        assert_array_equal(self.data['attr_month'], expected)

    def test_date_attribute(self):
        expected = np.array(['1999-01-31', '2004-12-01', '1817-04-28', '2100-09-10', '2013-11-30', '1631-10-15'], dtype='datetime64[D]')
        assert_array_equal(self.data['attr_date'], expected)

    def test_datetime_local_attribute(self):
        expected = np.array([datetime.datetime(year=1999, month=1, day=31, hour=0, minute=1), datetime.datetime(year=2004, month=12, day=1, hour=23, minute=59), datetime.datetime(year=1817, month=4, day=28, hour=13, minute=0), datetime.datetime(year=2100, month=9, day=10, hour=12, minute=0), datetime.datetime(year=2013, month=11, day=30, hour=4, minute=55), datetime.datetime(year=1631, month=10, day=15, hour=20, minute=4)], dtype='datetime64[m]')
        assert_array_equal(self.data['attr_datetime_local'], expected)

    def test_datetime_missing(self):
        expected = np.array(['nat', '2004-12-01T23:59', 'nat', 'nat', '2013-11-30T04:55', '1631-10-15T20:04'], dtype='datetime64[m]')
        assert_array_equal(self.data['attr_datetime_missing'], expected)

    def test_datetime_timezone(self):
        assert_raises(ParseArffError, loadarff, test8)