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
def _get_strcols_without_index(self) -> list[list[str]]:
    strcols: list[list[str]] = []
    if not is_list_like(self.header) and (not self.header):
        for i, c in enumerate(self.tr_frame):
            fmt_values = self.format_col(i)
            fmt_values = _make_fixed_width(strings=fmt_values, justify=self.justify, minimum=int(self.col_space.get(c, 0)), adj=self.adj)
            strcols.append(fmt_values)
        return strcols
    if is_list_like(self.header):
        self.header = cast(list[str], self.header)
        if len(self.header) != len(self.columns):
            raise ValueError(f'Writing {len(self.columns)} cols but got {len(self.header)} aliases')
        str_columns = [[label] for label in self.header]
    else:
        str_columns = self._get_formatted_column_labels(self.tr_frame)
    if self.show_row_idx_names:
        for x in str_columns:
            x.append('')
    for i, c in enumerate(self.tr_frame):
        cheader = str_columns[i]
        header_colwidth = max(int(self.col_space.get(c, 0)), *(self.adj.len(x) for x in cheader))
        fmt_values = self.format_col(i)
        fmt_values = _make_fixed_width(fmt_values, self.justify, minimum=header_colwidth, adj=self.adj)
        max_len = max(*(self.adj.len(x) for x in fmt_values), header_colwidth)
        cheader = self.adj.justify(cheader, max_len, mode=self.justify)
        strcols.append(cheader + fmt_values)
    return strcols