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
class DataFrameFormatter:
    """
    Class for processing dataframe formatting options and data.

    Used by DataFrame.to_string, which backs DataFrame.__repr__.
    """
    __doc__ = __doc__ if __doc__ else ''
    __doc__ += common_docstring + return_docstring

    def __init__(self, frame: DataFrame, columns: Axes | None=None, col_space: ColspaceArgType | None=None, header: bool | SequenceNotStr[str]=True, index: bool=True, na_rep: str='NaN', formatters: FormattersType | None=None, justify: str | None=None, float_format: FloatFormatType | None=None, sparsify: bool | None=None, index_names: bool=True, max_rows: int | None=None, min_rows: int | None=None, max_cols: int | None=None, show_dimensions: bool | str=False, decimal: str='.', bold_rows: bool=False, escape: bool=True) -> None:
        self.frame = frame
        self.columns = self._initialize_columns(columns)
        self.col_space = self._initialize_colspace(col_space)
        self.header = header
        self.index = index
        self.na_rep = na_rep
        self.formatters = self._initialize_formatters(formatters)
        self.justify = self._initialize_justify(justify)
        self.float_format = float_format
        self.sparsify = self._initialize_sparsify(sparsify)
        self.show_index_names = index_names
        self.decimal = decimal
        self.bold_rows = bold_rows
        self.escape = escape
        self.max_rows = max_rows
        self.min_rows = min_rows
        self.max_cols = max_cols
        self.show_dimensions = show_dimensions
        self.max_cols_fitted = self._calc_max_cols_fitted()
        self.max_rows_fitted = self._calc_max_rows_fitted()
        self.tr_frame = self.frame
        self.truncate()
        self.adj = printing.get_adjustment()

    def get_strcols(self) -> list[list[str]]:
        """
        Render a DataFrame to a list of columns (as lists of strings).
        """
        strcols = self._get_strcols_without_index()
        if self.index:
            str_index = self._get_formatted_index(self.tr_frame)
            strcols.insert(0, str_index)
        return strcols

    @property
    def should_show_dimensions(self) -> bool:
        return self.show_dimensions is True or (self.show_dimensions == 'truncate' and self.is_truncated)

    @property
    def is_truncated(self) -> bool:
        return bool(self.is_truncated_horizontally or self.is_truncated_vertically)

    @property
    def is_truncated_horizontally(self) -> bool:
        return bool(self.max_cols_fitted and len(self.columns) > self.max_cols_fitted)

    @property
    def is_truncated_vertically(self) -> bool:
        return bool(self.max_rows_fitted and len(self.frame) > self.max_rows_fitted)

    @property
    def dimensions_info(self) -> str:
        return f'\n\n[{len(self.frame)} rows x {len(self.frame.columns)} columns]'

    @property
    def has_index_names(self) -> bool:
        return _has_names(self.frame.index)

    @property
    def has_column_names(self) -> bool:
        return _has_names(self.frame.columns)

    @property
    def show_row_idx_names(self) -> bool:
        return all((self.has_index_names, self.index, self.show_index_names))

    @property
    def show_col_idx_names(self) -> bool:
        return all((self.has_column_names, self.show_index_names, self.header))

    @property
    def max_rows_displayed(self) -> int:
        return min(self.max_rows or len(self.frame), len(self.frame))

    def _initialize_sparsify(self, sparsify: bool | None) -> bool:
        if sparsify is None:
            return get_option('display.multi_sparse')
        return sparsify

    def _initialize_formatters(self, formatters: FormattersType | None) -> FormattersType:
        if formatters is None:
            return {}
        elif len(self.frame.columns) == len(formatters) or isinstance(formatters, dict):
            return formatters
        else:
            raise ValueError(f'Formatters length({len(formatters)}) should match DataFrame number of columns({len(self.frame.columns)})')

    def _initialize_justify(self, justify: str | None) -> str:
        if justify is None:
            return get_option('display.colheader_justify')
        else:
            return justify

    def _initialize_columns(self, columns: Axes | None) -> Index:
        if columns is not None:
            cols = ensure_index(columns)
            self.frame = self.frame[cols]
            return cols
        else:
            return self.frame.columns

    def _initialize_colspace(self, col_space: ColspaceArgType | None) -> ColspaceType:
        result: ColspaceType
        if col_space is None:
            result = {}
        elif isinstance(col_space, (int, str)):
            result = {'': col_space}
            result.update({column: col_space for column in self.frame.columns})
        elif isinstance(col_space, Mapping):
            for column in col_space.keys():
                if column not in self.frame.columns and column != '':
                    raise ValueError(f'Col_space is defined for an unknown column: {column}')
            result = col_space
        else:
            if len(self.frame.columns) != len(col_space):
                raise ValueError(f'Col_space length({len(col_space)}) should match DataFrame number of columns({len(self.frame.columns)})')
            result = dict(zip(self.frame.columns, col_space))
        return result

    def _calc_max_cols_fitted(self) -> int | None:
        """Number of columns fitting the screen."""
        if not self._is_in_terminal():
            return self.max_cols
        width, _ = get_terminal_size()
        if self._is_screen_narrow(width):
            return width
        else:
            return self.max_cols

    def _calc_max_rows_fitted(self) -> int | None:
        """Number of rows with data fitting the screen."""
        max_rows: int | None
        if self._is_in_terminal():
            _, height = get_terminal_size()
            if self.max_rows == 0:
                return height - self._get_number_of_auxiliary_rows()
            if self._is_screen_short(height):
                max_rows = height
            else:
                max_rows = self.max_rows
        else:
            max_rows = self.max_rows
        return self._adjust_max_rows(max_rows)

    def _adjust_max_rows(self, max_rows: int | None) -> int | None:
        """Adjust max_rows using display logic.

        See description here:
        https://pandas.pydata.org/docs/dev/user_guide/options.html#frequently-used-options

        GH #37359
        """
        if max_rows:
            if len(self.frame) > max_rows and self.min_rows:
                max_rows = min(self.min_rows, max_rows)
        return max_rows

    def _is_in_terminal(self) -> bool:
        """Check if the output is to be shown in terminal."""
        return bool(self.max_cols == 0 or self.max_rows == 0)

    def _is_screen_narrow(self, max_width) -> bool:
        return bool(self.max_cols == 0 and len(self.frame.columns) > max_width)

    def _is_screen_short(self, max_height) -> bool:
        return bool(self.max_rows == 0 and len(self.frame) > max_height)

    def _get_number_of_auxiliary_rows(self) -> int:
        """Get number of rows occupied by prompt, dots and dimension info."""
        dot_row = 1
        prompt_row = 1
        num_rows = dot_row + prompt_row
        if self.show_dimensions:
            num_rows += len(self.dimensions_info.splitlines())
        if self.header:
            num_rows += 1
        return num_rows

    def truncate(self) -> None:
        """
        Check whether the frame should be truncated. If so, slice the frame up.
        """
        if self.is_truncated_horizontally:
            self._truncate_horizontally()
        if self.is_truncated_vertically:
            self._truncate_vertically()

    def _truncate_horizontally(self) -> None:
        """Remove columns, which are not to be displayed and adjust formatters.

        Attributes affected:
            - tr_frame
            - formatters
            - tr_col_num
        """
        assert self.max_cols_fitted is not None
        col_num = self.max_cols_fitted // 2
        if col_num >= 1:
            left = self.tr_frame.iloc[:, :col_num]
            right = self.tr_frame.iloc[:, -col_num:]
            self.tr_frame = concat((left, right), axis=1)
            if isinstance(self.formatters, (list, tuple)):
                self.formatters = [*self.formatters[:col_num], *self.formatters[-col_num:]]
        else:
            col_num = cast(int, self.max_cols)
            self.tr_frame = self.tr_frame.iloc[:, :col_num]
        self.tr_col_num = col_num

    def _truncate_vertically(self) -> None:
        """Remove rows, which are not to be displayed.

        Attributes affected:
            - tr_frame
            - tr_row_num
        """
        assert self.max_rows_fitted is not None
        row_num = self.max_rows_fitted // 2
        if row_num >= 1:
            _len = len(self.tr_frame)
            _slice = np.hstack([np.arange(row_num), np.arange(_len - row_num, _len)])
            self.tr_frame = self.tr_frame.iloc[_slice]
        else:
            row_num = cast(int, self.max_rows)
            self.tr_frame = self.tr_frame.iloc[:row_num, :]
        self.tr_row_num = row_num

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

    def format_col(self, i: int) -> list[str]:
        frame = self.tr_frame
        formatter = self._get_formatter(i)
        return format_array(frame.iloc[:, i]._values, formatter, float_format=self.float_format, na_rep=self.na_rep, space=self.col_space.get(frame.columns[i]), decimal=self.decimal, leading_space=self.index)

    def _get_formatter(self, i: str | int) -> Callable | None:
        if isinstance(self.formatters, (list, tuple)):
            if is_integer(i):
                i = cast(int, i)
                return self.formatters[i]
            else:
                return None
        else:
            if is_integer(i) and i not in self.columns:
                i = self.columns[i]
            return self.formatters.get(i, None)

    def _get_formatted_column_labels(self, frame: DataFrame) -> list[list[str]]:
        from pandas.core.indexes.multi import sparsify_labels
        columns = frame.columns
        if isinstance(columns, MultiIndex):
            fmt_columns = columns._format_multi(sparsify=False, include_names=False)
            fmt_columns = list(zip(*fmt_columns))
            dtypes = self.frame.dtypes._values
            restrict_formatting = any((level.is_floating for level in columns.levels))
            need_leadsp = dict(zip(fmt_columns, map(is_numeric_dtype, dtypes)))

            def space_format(x, y):
                if y not in self.formatters and need_leadsp[x] and (not restrict_formatting):
                    return ' ' + y
                return y
            str_columns_tuple = list(zip(*([space_format(x, y) for y in x] for x in fmt_columns)))
            if self.sparsify and len(str_columns_tuple):
                str_columns_tuple = sparsify_labels(str_columns_tuple)
            str_columns = [list(x) for x in zip(*str_columns_tuple)]
        else:
            fmt_columns = columns._format_flat(include_name=False)
            dtypes = self.frame.dtypes
            need_leadsp = dict(zip(fmt_columns, map(is_numeric_dtype, dtypes)))
            str_columns = [[' ' + x if not self._get_formatter(i) and need_leadsp[x] else x] for i, x in enumerate(fmt_columns)]
        return str_columns

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

    def _get_column_name_list(self) -> list[Hashable]:
        names: list[Hashable] = []
        columns = self.frame.columns
        if isinstance(columns, MultiIndex):
            names.extend(('' if name is None else name for name in columns.names))
        else:
            names.append('' if columns.name is None else columns.name)
        return names