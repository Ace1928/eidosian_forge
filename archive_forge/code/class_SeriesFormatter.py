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
class SeriesFormatter:
    """
    Implement the main logic of Series.to_string, which underlies
    Series.__repr__.
    """

    def __init__(self, series: Series, *, length: bool | str=True, header: bool=True, index: bool=True, na_rep: str='NaN', name: bool=False, float_format: str | None=None, dtype: bool=True, max_rows: int | None=None, min_rows: int | None=None) -> None:
        self.series = series
        self.buf = StringIO()
        self.name = name
        self.na_rep = na_rep
        self.header = header
        self.length = length
        self.index = index
        self.max_rows = max_rows
        self.min_rows = min_rows
        if float_format is None:
            float_format = get_option('display.float_format')
        self.float_format = float_format
        self.dtype = dtype
        self.adj = printing.get_adjustment()
        self._chk_truncate()

    def _chk_truncate(self) -> None:
        self.tr_row_num: int | None
        min_rows = self.min_rows
        max_rows = self.max_rows
        is_truncated_vertically = max_rows and len(self.series) > max_rows
        series = self.series
        if is_truncated_vertically:
            max_rows = cast(int, max_rows)
            if min_rows:
                max_rows = min(min_rows, max_rows)
            if max_rows == 1:
                row_num = max_rows
                series = series.iloc[:max_rows]
            else:
                row_num = max_rows // 2
                series = concat((series.iloc[:row_num], series.iloc[-row_num:]))
            self.tr_row_num = row_num
        else:
            self.tr_row_num = None
        self.tr_series = series
        self.is_truncated_vertically = is_truncated_vertically

    def _get_footer(self) -> str:
        name = self.series.name
        footer = ''
        index = self.series.index
        if isinstance(index, (DatetimeIndex, PeriodIndex, TimedeltaIndex)) and index.freq is not None:
            footer += f'Freq: {index.freqstr}'
        if self.name is not False and name is not None:
            if footer:
                footer += ', '
            series_name = printing.pprint_thing(name, escape_chars=('\t', '\r', '\n'))
            footer += f'Name: {series_name}'
        if self.length is True or (self.length == 'truncate' and self.is_truncated_vertically):
            if footer:
                footer += ', '
            footer += f'Length: {len(self.series)}'
        if self.dtype is not False and self.dtype is not None:
            dtype_name = getattr(self.tr_series.dtype, 'name', None)
            if dtype_name:
                if footer:
                    footer += ', '
                footer += f'dtype: {printing.pprint_thing(dtype_name)}'
        if isinstance(self.tr_series.dtype, CategoricalDtype):
            level_info = self.tr_series._values._get_repr_footer()
            if footer:
                footer += '\n'
            footer += level_info
        return str(footer)

    def _get_formatted_values(self) -> list[str]:
        return format_array(self.tr_series._values, None, float_format=self.float_format, na_rep=self.na_rep, leading_space=self.index)

    def to_string(self) -> str:
        series = self.tr_series
        footer = self._get_footer()
        if len(series) == 0:
            return f'{type(self.series).__name__}([], {footer})'
        index = series.index
        have_header = _has_names(index)
        if isinstance(index, MultiIndex):
            fmt_index = index._format_multi(include_names=True, sparsify=None)
            adj = printing.get_adjustment()
            fmt_index = adj.adjoin(2, *fmt_index).split('\n')
        else:
            fmt_index = index._format_flat(include_name=True)
        fmt_values = self._get_formatted_values()
        if self.is_truncated_vertically:
            n_header_rows = 0
            row_num = self.tr_row_num
            row_num = cast(int, row_num)
            width = self.adj.len(fmt_values[row_num - 1])
            if width > 3:
                dot_str = '...'
            else:
                dot_str = '..'
            dot_str = self.adj.justify([dot_str], width, mode='center')[0]
            fmt_values.insert(row_num + n_header_rows, dot_str)
            fmt_index.insert(row_num + 1, '')
        if self.index:
            result = self.adj.adjoin(3, *[fmt_index[1:], fmt_values])
        else:
            result = self.adj.adjoin(3, fmt_values)
        if self.header and have_header:
            result = fmt_index[0] + '\n' + result
        if footer:
            result += '\n' + footer
        return str(''.join(result))