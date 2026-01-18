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
class TestFormatPercentiles:

    @pytest.mark.parametrize('percentiles, expected', [([0.01999, 0.02001, 0.5, 0.666666, 0.9999], ['1.999%', '2.001%', '50%', '66.667%', '99.99%']), ([0, 0.5, 0.02001, 0.5, 0.666666, 0.9999], ['0%', '50%', '2.0%', '50%', '66.67%', '99.99%']), ([0.281, 0.29, 0.57, 0.58], ['28.1%', '29%', '57%', '58%']), ([0.28, 0.29, 0.57, 0.58], ['28%', '29%', '57%', '58%']), ([0.9, 0.99, 0.999, 0.9999, 0.99999], ['90%', '99%', '99.9%', '99.99%', '99.999%'])])
    def test_format_percentiles(self, percentiles, expected):
        result = fmt.format_percentiles(percentiles)
        assert result == expected

    @pytest.mark.parametrize('percentiles', [[0.1, np.nan, 0.5], [-0.001, 0.1, 0.5], [2, 0.1, 0.5], [0.1, 0.5, 'a']])
    def test_error_format_percentiles(self, percentiles):
        msg = 'percentiles should all be in the interval \\[0,1\\]'
        with pytest.raises(ValueError, match=msg):
            fmt.format_percentiles(percentiles)

    def test_format_percentiles_integer_idx(self):
        result = fmt.format_percentiles(np.linspace(0, 1, 10 + 1))
        expected = ['0%', '10%', '20%', '30%', '40%', '50%', '60%', '70%', '80%', '90%', '100%']
        assert result == expected