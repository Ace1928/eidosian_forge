import pytest
import textwrap
from numpy.testing import assert_array_equal, assert_equal, assert_raises
import numpy as np
from numpy.f2py.tests import util
class TestStringScalarArr(util.F2PyTest):
    sources = [util.getpath('tests', 'src', 'string', 'scalar_string.f90')]

    def test_char(self):
        for out in (self.module.string_test.string, self.module.string_test.string77):
            expected = ()
            assert out.shape == expected
            expected = '|S8'
            assert out.dtype == expected

    def test_char_arr(self):
        for out in (self.module.string_test.strarr, self.module.string_test.strarr77):
            expected = (5, 7)
            assert out.shape == expected
            expected = '|S12'
            assert out.dtype == expected