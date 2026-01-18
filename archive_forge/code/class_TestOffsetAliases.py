from __future__ import annotations
from datetime import (
import numpy as np
import pytest
from pandas._libs.tslibs import (
import pandas._libs.tslibs.offsets as liboffsets
from pandas._libs.tslibs.offsets import (
from pandas._libs.tslibs.period import INVALID_FREQ_ERR_MSG
from pandas.errors import PerformanceWarning
from pandas import (
import pandas._testing as tm
from pandas.tests.tseries.offsets.common import WeekDay
from pandas.tseries import offsets
from pandas.tseries.offsets import (
class TestOffsetAliases:

    def setup_method(self):
        _offset_map.clear()

    def test_alias_equality(self):
        for k, v in _offset_map.items():
            if v is None:
                continue
            assert k == v.copy()

    def test_rule_code(self):
        lst = ['ME', 'MS', 'BME', 'BMS', 'D', 'B', 'h', 'min', 's', 'ms', 'us']
        for k in lst:
            assert k == _get_offset(k).rule_code
            assert k in _offset_map
            assert k == (_get_offset(k) * 3).rule_code
        suffix_lst = ['MON', 'TUE', 'WED', 'THU', 'FRI', 'SAT', 'SUN']
        base = 'W'
        for v in suffix_lst:
            alias = '-'.join([base, v])
            assert alias == _get_offset(alias).rule_code
            assert alias == (_get_offset(alias) * 5).rule_code
        suffix_lst = ['JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 'JUL', 'AUG', 'SEP', 'OCT', 'NOV', 'DEC']
        base_lst = ['YE', 'YS', 'BYE', 'BYS', 'QE', 'QS', 'BQE', 'BQS']
        for base in base_lst:
            for v in suffix_lst:
                alias = '-'.join([base, v])
                assert alias == _get_offset(alias).rule_code
                assert alias == (_get_offset(alias) * 5).rule_code