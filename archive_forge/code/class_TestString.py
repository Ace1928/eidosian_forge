import os
import pytest
import textwrap
import numpy as np
from . import util
class TestString(util.F2PyTest):
    sources = [util.getpath('tests', 'src', 'string', 'char.f90')]

    @pytest.mark.slow
    def test_char(self):
        strings = np.array(['ab', 'cd', 'ef'], dtype='c').T
        inp, out = self.module.char_test.change_strings(strings, strings.shape[1])
        assert inp == pytest.approx(strings)
        expected = strings.copy()
        expected[1, :] = 'AAA'
        assert out == pytest.approx(expected)