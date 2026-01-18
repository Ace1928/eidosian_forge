from __future__ import annotations
from collections.abc import (
from contextlib import contextmanager
from csv import QUOTE_NONE
from decimal import Decimal
from functools import partial
from io import StringIO
import math
import re
from shutil import get_terminal_size
from typing import (
import numpy as np
from pandas._config.config import (
from pandas._libs import lib
from pandas._libs.missing import NA
from pandas._libs.tslibs import (
from pandas._libs.tslibs.nattype import NaTType
from pandas.core.dtypes.common import (
from pandas.core.dtypes.dtypes import (
from pandas.core.dtypes.missing import (
from pandas.core.arrays import (
from pandas.core.arrays.string_ import StringDtype
from pandas.core.base import PandasObject
import pandas.core.common as com
from pandas.core.indexes.api import (
from pandas.core.indexes.datetimes import DatetimeIndex
from pandas.core.indexes.timedeltas import TimedeltaIndex
from pandas.core.reshape.concat import concat
from pandas.io.common import (
from pandas.io.formats import printing
def _get_formatted_index(self, frame: DataFrame) -> list[str]:
    col_space = {k: cast(int, v) for k, v in self.col_space.items()}
    index = frame.index
    columns = frame.columns
    fmt = self._get_formatter('__index__')
    if isinstance(index, MultiIndex):
        fmt_index = index._format_multi(sparsify=self.sparsify, include_names=self.show_row_idx_names, formatter=fmt)
    else:
        fmt_index = [index._format_flat(include_name=self.show_row_idx_names, formatter=fmt)]
    fmt_index = [tuple(_make_fixed_width(list(x), justify='left', minimum=col_space.get('', 0), adj=self.adj)) for x in fmt_index]
    adjoined = self.adj.adjoin(1, *fmt_index).split('\n')
    if self.show_col_idx_names:
        col_header = [str(x) for x in self._get_column_name_list()]
    else:
        col_header = [''] * columns.nlevels
    if self.header:
        return col_header + adjoined
    else:
        return adjoined