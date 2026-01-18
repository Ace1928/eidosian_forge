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
class TestMissingData:

    def test_missing(self):
        data, meta = loadarff(missing)
        for i in ['yop', 'yap']:
            assert_array_almost_equal(data[i], expect_missing[i])