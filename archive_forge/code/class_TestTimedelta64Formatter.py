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
class TestTimedelta64Formatter:

    def test_days(self):
        x = pd.to_timedelta(list(range(5)) + [NaT], unit='D')._values
        result = fmt._Timedelta64Formatter(x).get_result()
        assert result[0].strip() == '0 days'
        assert result[1].strip() == '1 days'
        result = fmt._Timedelta64Formatter(x[1:2]).get_result()
        assert result[0].strip() == '1 days'
        result = fmt._Timedelta64Formatter(x).get_result()
        assert result[0].strip() == '0 days'
        assert result[1].strip() == '1 days'
        result = fmt._Timedelta64Formatter(x[1:2]).get_result()
        assert result[0].strip() == '1 days'

    def test_days_neg(self):
        x = pd.to_timedelta(list(range(5)) + [NaT], unit='D')._values
        result = fmt._Timedelta64Formatter(-x).get_result()
        assert result[0].strip() == '0 days'
        assert result[1].strip() == '-1 days'

    def test_subdays(self):
        y = pd.to_timedelta(list(range(5)) + [NaT], unit='s')._values
        result = fmt._Timedelta64Formatter(y).get_result()
        assert result[0].strip() == '0 days 00:00:00'
        assert result[1].strip() == '0 days 00:00:01'

    def test_subdays_neg(self):
        y = pd.to_timedelta(list(range(5)) + [NaT], unit='s')._values
        result = fmt._Timedelta64Formatter(-y).get_result()
        assert result[0].strip() == '0 days 00:00:00'
        assert result[1].strip() == '-1 days +23:59:59'

    def test_zero(self):
        x = pd.to_timedelta(list(range(1)) + [NaT], unit='D')._values
        result = fmt._Timedelta64Formatter(x).get_result()
        assert result[0].strip() == '0 days'
        x = pd.to_timedelta(list(range(1)), unit='D')._values
        result = fmt._Timedelta64Formatter(x).get_result()
        assert result[0].strip() == '0 days'