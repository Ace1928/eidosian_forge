from datetime import datetime
from io import StringIO
from pathlib import Path
import re
from shutil import get_terminal_size
import numpy as np
import pytest
from pandas._config import using_pyarrow_string_dtype
import pandas as pd
from pandas import (
from pandas.io.formats import printing
import pandas.io.formats.format as fmt
class TestFloatArrayFormatter:

    def test_misc(self):
        obj = fmt.FloatArrayFormatter(np.array([], dtype=np.float64))
        result = obj.get_result()
        assert len(result) == 0

    def test_format(self):
        obj = fmt.FloatArrayFormatter(np.array([12, 0], dtype=np.float64))
        result = obj.get_result()
        assert result[0] == ' 12.0'
        assert result[1] == '  0.0'

    def test_output_display_precision_trailing_zeroes(self):
        with option_context('display.precision', 0):
            s = Series([840.0, 4200.0])
            expected_output = '0     840\n1    4200\ndtype: float64'
            assert str(s) == expected_output

    @pytest.mark.parametrize('value,expected', [([9.4444], '   0\n0  9'), ([0.49], '       0\n0  5e-01'), ([10.9999], '    0\n0  11'), ([9.5444, 9.6], '    0\n0  10\n1  10'), ([0.46, 0.78, -9.9999], '       0\n0  5e-01\n1  8e-01\n2 -1e+01')])
    def test_set_option_precision(self, value, expected):
        with option_context('display.precision', 0):
            df_value = DataFrame(value)
            assert str(df_value) == expected

    def test_output_significant_digits(self):
        with option_context('display.precision', 6):
            d = DataFrame({'col1': [9.999e-08, 1e-07, 1.0001e-07, 2e-07, 4.999e-07, 5e-07, 5.0001e-07, 6e-07, 9.999e-07, 1e-06, 1.0001e-06, 2e-06, 4.999e-06, 5e-06, 5.0001e-06, 6e-06]})
            expected_output = {(0, 6): '           col1\n0  9.999000e-08\n1  1.000000e-07\n2  1.000100e-07\n3  2.000000e-07\n4  4.999000e-07\n5  5.000000e-07', (1, 6): '           col1\n1  1.000000e-07\n2  1.000100e-07\n3  2.000000e-07\n4  4.999000e-07\n5  5.000000e-07', (1, 8): '           col1\n1  1.000000e-07\n2  1.000100e-07\n3  2.000000e-07\n4  4.999000e-07\n5  5.000000e-07\n6  5.000100e-07\n7  6.000000e-07', (8, 16): '            col1\n8   9.999000e-07\n9   1.000000e-06\n10  1.000100e-06\n11  2.000000e-06\n12  4.999000e-06\n13  5.000000e-06\n14  5.000100e-06\n15  6.000000e-06', (9, 16): '        col1\n9   0.000001\n10  0.000001\n11  0.000002\n12  0.000005\n13  0.000005\n14  0.000005\n15  0.000006'}
            for (start, stop), v in expected_output.items():
                assert str(d[start:stop]) == v

    def test_too_long(self):
        with option_context('display.precision', 4):
            df = DataFrame({'x': [12345.6789]})
            assert str(df) == '            x\n0  12345.6789'
            df = DataFrame({'x': [2000000.0]})
            assert str(df) == '           x\n0  2000000.0'
            df = DataFrame({'x': [12345.6789, 2000000.0]})
            assert str(df) == '            x\n0  1.2346e+04\n1  2.0000e+06'