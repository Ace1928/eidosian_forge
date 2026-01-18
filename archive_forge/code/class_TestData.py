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
class TestData:

    def test1(self):
        self._test(test4)

    def test2(self):
        self._test(test5)

    def test3(self):
        self._test(test6)

    def test4(self):
        self._test(test11)

    def _test(self, test_file):
        data, meta = loadarff(test_file)
        for i in range(len(data)):
            for j in range(4):
                assert_array_almost_equal(expect4_data[i][j], data[i][j])
        assert_equal(meta.types(), expected_types)

    def test_filelike(self):
        with open(test1) as f1:
            data1, meta1 = loadarff(f1)
        with open(test1) as f2:
            data2, meta2 = loadarff(StringIO(f2.read()))
        assert_(data1 == data2)
        assert_(repr(meta1) == repr(meta2))

    def test_path(self):
        from pathlib import Path
        with open(test1) as f1:
            data1, meta1 = loadarff(f1)
        data2, meta2 = loadarff(Path(test1))
        assert_(data1 == data2)
        assert_(repr(meta1) == repr(meta2))