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
class TestQuotedNominalSpaces:
    """
    Regression test for issue #10232:
    
    Exception in loadarff with quoted nominal attributes.
    """

    def setup_method(self):
        self.data, self.meta = loadarff(test_quoted_nominal_spaces)

    def test_attributes(self):
        assert_equal(len(self.meta._attributes), 2)
        age, smoker = self.meta._attributes.values()
        assert_equal(age.name, 'age')
        assert_equal(age.type_name, 'numeric')
        assert_equal(smoker.name, 'smoker')
        assert_equal(smoker.type_name, 'nominal')
        assert_equal(smoker.values, ['  yes', 'no  '])

    def test_data(self):
        age_dtype_instance = np.float64
        smoker_dtype_instance = '<S5'
        age_expected = np.array([18, 24, 44, 56, 89, 11], dtype=age_dtype_instance)
        smoker_expected = np.array(['no  ', '  yes', 'no  ', 'no  ', '  yes', 'no  '], dtype=smoker_dtype_instance)
        assert_array_equal(self.data['age'], age_expected)
        assert_array_equal(self.data['smoker'], smoker_expected)