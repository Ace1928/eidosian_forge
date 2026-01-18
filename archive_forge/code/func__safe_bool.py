from abc import ABC, abstractmethod
from builtins import bool
from typing import Any, Dict, List, Tuple
from triad.utils.assertion import assert_or_throw
from triad.utils.pandas_like import PandasLikeUtils
from qpd.dataframe import Column, DataFrame
from qpd.specs import (
def _safe_bool(self, col: Any) -> Any:
    if self.is_series(col):
        return col.astype('f8')
    if self.is_series(col.native):
        return col.native.astype('f8')
    elif col is None:
        return float('nan')
    else:
        return float(col.native > 0)