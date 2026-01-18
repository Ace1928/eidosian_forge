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
class TestRelationalAttribute:

    def setup_method(self):
        self.data, self.meta = loadarff(test9)

    def test_attributes(self):
        assert_equal(len(self.meta._attributes), 1)
        relational = list(self.meta._attributes.values())[0]
        assert_equal(relational.name, 'attr_date_number')
        assert_equal(relational.type_name, 'relational')
        assert_equal(len(relational.attributes), 2)
        assert_equal(relational.attributes[0].name, 'attr_date')
        assert_equal(relational.attributes[0].type_name, 'date')
        assert_equal(relational.attributes[1].name, 'attr_number')
        assert_equal(relational.attributes[1].type_name, 'numeric')

    def test_data(self):
        dtype_instance = [('attr_date', 'datetime64[D]'), ('attr_number', np.float64)]
        expected = [np.array([('1999-01-31', 1), ('1935-11-27', 10)], dtype=dtype_instance), np.array([('2004-12-01', 2), ('1942-08-13', 20)], dtype=dtype_instance), np.array([('1817-04-28', 3)], dtype=dtype_instance), np.array([('2100-09-10', 4), ('1957-04-17', 40), ('1721-01-14', 400)], dtype=dtype_instance), np.array([('2013-11-30', 5)], dtype=dtype_instance), np.array([('1631-10-15', 6)], dtype=dtype_instance)]
        for i in range(len(self.data['attr_date_number'])):
            assert_array_equal(self.data['attr_date_number'][i], expected[i])