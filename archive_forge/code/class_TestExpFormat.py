import numpy as np
from numpy.testing import assert_equal
from pytest import raises as assert_raises
from scipy.io._harwell_boeing import (
class TestExpFormat:

    def test_to_fortran(self):
        f = [ExpFormat(10, 5), ExpFormat(12, 10), ExpFormat(12, 10, min=3), ExpFormat(10, 5, repeat=3)]
        res = ['(E10.5)', '(E12.10)', '(E12.10E3)', '(3E10.5)']
        for i, j in zip(f, res):
            assert_equal(i.fortran_format, j)

    def test_from_number(self):
        f = np.array([1.0, -1.2])
        r_f = [ExpFormat(24, 16, repeat=3), ExpFormat(25, 16, repeat=3)]
        for i, j in zip(f, r_f):
            assert_equal(ExpFormat.from_number(i).__dict__, j.__dict__)