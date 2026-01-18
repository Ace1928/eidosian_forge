from __future__ import annotations
from textwrap import dedent
from typing import (
from pandas._config import get_option
from pandas._libs import lib
from pandas import (
from pandas.io.common import is_url
from pandas.io.formats.format import (
from pandas.io.formats.printing import pprint_thing
def _write_regular_rows(self, fmt_values: Mapping[int, list[str]], indent: int) -> None:
    is_truncated_horizontally = self.fmt.is_truncated_horizontally
    is_truncated_vertically = self.fmt.is_truncated_vertically
    nrows = len(self.fmt.tr_frame)
    if self.fmt.index:
        fmt = self.fmt._get_formatter('__index__')
        if fmt is not None:
            index_values = self.fmt.tr_frame.index.map(fmt)
        else:
            index_values = self.fmt.tr_frame.index._format_flat(include_name=False)
    row: list[str] = []
    for i in range(nrows):
        if is_truncated_vertically and i == self.fmt.tr_row_num:
            str_sep_row = ['...'] * len(row)
            self.write_tr(str_sep_row, indent, self.indent_delta, tags=None, nindex_levels=self.row_levels)
        row = []
        if self.fmt.index:
            row.append(index_values[i])
        elif self.show_col_idx_names:
            row.append('')
        row.extend((fmt_values[j][i] for j in range(self.ncols)))
        if is_truncated_horizontally:
            dot_col_ix = self.fmt.tr_col_num + self.row_levels
            row.insert(dot_col_ix, '...')
        self.write_tr(row, indent, self.indent_delta, tags=None, nindex_levels=self.row_levels)