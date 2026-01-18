from __future__ import annotations
from collections import (
from collections.abc import (
import csv
from io import StringIO
import re
from typing import (
import warnings
import numpy as np
from pandas._libs import lib
from pandas.errors import (
from pandas.util._decorators import cache_readonly
from pandas.util._exceptions import find_stack_level
from pandas.core.dtypes.common import (
from pandas.core.dtypes.inference import is_dict_like
from pandas.io.common import (
from pandas.io.parsers.base_parser import (
def _infer_columns(self) -> tuple[list[list[Scalar | None]], int, set[Scalar | None]]:
    names = self.names
    num_original_columns = 0
    clear_buffer = True
    unnamed_cols: set[Scalar | None] = set()
    if self.header is not None:
        header = self.header
        have_mi_columns = self._have_mi_columns
        if isinstance(header, (list, tuple, np.ndarray)):
            if have_mi_columns:
                header = list(header) + [header[-1] + 1]
        else:
            header = [header]
        columns: list[list[Scalar | None]] = []
        for level, hr in enumerate(header):
            try:
                line = self._buffered_line()
                while self.line_pos <= hr:
                    line = self._next_line()
            except StopIteration as err:
                if 0 < self.line_pos <= hr and (not have_mi_columns or hr != header[-1]):
                    joi = list(map(str, header[:-1] if have_mi_columns else header))
                    msg = f'[{','.join(joi)}], len of {len(joi)}, '
                    raise ValueError(f'Passed header={msg}but only {self.line_pos} lines in file') from err
                if have_mi_columns and hr > 0:
                    if clear_buffer:
                        self._clear_buffer()
                    columns.append([None] * len(columns[-1]))
                    return (columns, num_original_columns, unnamed_cols)
                if not self.names:
                    raise EmptyDataError('No columns to parse from file') from err
                line = self.names[:]
            this_columns: list[Scalar | None] = []
            this_unnamed_cols = []
            for i, c in enumerate(line):
                if c == '':
                    if have_mi_columns:
                        col_name = f'Unnamed: {i}_level_{level}'
                    else:
                        col_name = f'Unnamed: {i}'
                    this_unnamed_cols.append(i)
                    this_columns.append(col_name)
                else:
                    this_columns.append(c)
            if not have_mi_columns:
                counts: DefaultDict = defaultdict(int)
                col_loop_order = [i for i in range(len(this_columns)) if i not in this_unnamed_cols] + this_unnamed_cols
                for i in col_loop_order:
                    col = this_columns[i]
                    old_col = col
                    cur_count = counts[col]
                    if cur_count > 0:
                        while cur_count > 0:
                            counts[old_col] = cur_count + 1
                            col = f'{old_col}.{cur_count}'
                            if col in this_columns:
                                cur_count += 1
                            else:
                                cur_count = counts[col]
                        if self.dtype is not None and is_dict_like(self.dtype) and (self.dtype.get(old_col) is not None) and (self.dtype.get(col) is None):
                            self.dtype.update({col: self.dtype.get(old_col)})
                    this_columns[i] = col
                    counts[col] = cur_count + 1
            elif have_mi_columns:
                if hr == header[-1]:
                    lc = len(this_columns)
                    sic = self.index_col
                    ic = len(sic) if sic is not None else 0
                    unnamed_count = len(this_unnamed_cols)
                    if lc != unnamed_count and lc - ic > unnamed_count or ic == 0:
                        clear_buffer = False
                        this_columns = [None] * lc
                        self.buf = [self.buf[-1]]
            columns.append(this_columns)
            unnamed_cols.update({this_columns[i] for i in this_unnamed_cols})
            if len(columns) == 1:
                num_original_columns = len(this_columns)
        if clear_buffer:
            self._clear_buffer()
        first_line: list[Scalar] | None
        if names is not None:
            try:
                first_line = self._next_line()
            except StopIteration:
                first_line = None
            len_first_data_row = 0 if first_line is None else len(first_line)
            if len(names) > len(columns[0]) and len(names) > len_first_data_row:
                raise ValueError('Number of passed names did not match number of header fields in the file')
            if len(columns) > 1:
                raise TypeError('Cannot pass names with multi-index columns')
            if self.usecols is not None:
                self._handle_usecols(columns, names, num_original_columns)
            else:
                num_original_columns = len(names)
            if self._col_indices is not None and len(names) != len(self._col_indices):
                columns = [[names[i] for i in sorted(self._col_indices)]]
            else:
                columns = [names]
        else:
            columns = self._handle_usecols(columns, columns[0], num_original_columns)
    else:
        ncols = len(self._header_line)
        num_original_columns = ncols
        if not names:
            columns = [list(range(ncols))]
            columns = self._handle_usecols(columns, columns[0], ncols)
        elif self.usecols is None or len(names) >= ncols:
            columns = self._handle_usecols([names], names, ncols)
            num_original_columns = len(names)
        elif not callable(self.usecols) and len(names) != len(self.usecols):
            raise ValueError('Number of passed names did not match number of header fields in the file')
        else:
            columns = [names]
            self._handle_usecols(columns, columns[0], ncols)
    return (columns, num_original_columns, unnamed_cols)