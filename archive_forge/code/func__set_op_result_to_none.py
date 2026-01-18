from abc import ABC, abstractmethod
from builtins import bool
from typing import Any, Dict, List, Tuple
from triad.utils.assertion import assert_or_throw
from triad.utils.pandas_like import PandasLikeUtils
from qpd.dataframe import Column, DataFrame
from qpd.specs import (
def _set_op_result_to_none(self, series: Any, s1: Any, s2: Any) -> Column:
    if not self.is_series(series):
        if s1 is None or s2 is None:
            return Column(None)
        return Column(series)
    if self.is_series(s1):
        series = series.mask(s1.isnull(), None)
    if self.is_series(s2):
        series = series.mask(s2.isnull(), None)
    return Column(series)