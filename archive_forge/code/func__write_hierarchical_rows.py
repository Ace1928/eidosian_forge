from __future__ import annotations
from textwrap import dedent
from typing import (
from pandas._config import get_option
from pandas._libs import lib
from pandas import (
from pandas.io.common import is_url
from pandas.io.formats.format import (
from pandas.io.formats.printing import pprint_thing
def _write_hierarchical_rows(self, fmt_values: Mapping[int, list[str]], indent: int) -> None:
    template = 'rowspan="{span}" valign="top"'
    is_truncated_horizontally = self.fmt.is_truncated_horizontally
    is_truncated_vertically = self.fmt.is_truncated_vertically
    frame = self.fmt.tr_frame
    nrows = len(frame)
    assert isinstance(frame.index, MultiIndex)
    idx_values = frame.index._format_multi(sparsify=False, include_names=False)
    idx_values = list(zip(*idx_values))
    if self.fmt.sparsify:
        sentinel = lib.no_default
        levels = frame.index._format_multi(sparsify=sentinel, include_names=False)
        level_lengths = get_level_lengths(levels, sentinel)
        inner_lvl = len(level_lengths) - 1
        if is_truncated_vertically:
            ins_row = self.fmt.tr_row_num
            inserted = False
            for lnum, records in enumerate(level_lengths):
                rec_new = {}
                for tag, span in list(records.items()):
                    if tag >= ins_row:
                        rec_new[tag + 1] = span
                    elif tag + span > ins_row:
                        rec_new[tag] = span + 1
                        if not inserted:
                            dot_row = list(idx_values[ins_row - 1])
                            dot_row[-1] = '...'
                            idx_values.insert(ins_row, tuple(dot_row))
                            inserted = True
                        else:
                            dot_row = list(idx_values[ins_row])
                            dot_row[inner_lvl - lnum] = '...'
                            idx_values[ins_row] = tuple(dot_row)
                    else:
                        rec_new[tag] = span
                    if tag + span == ins_row:
                        rec_new[ins_row] = 1
                        if lnum == 0:
                            idx_values.insert(ins_row, tuple(['...'] * len(level_lengths)))
                        elif inserted:
                            dot_row = list(idx_values[ins_row])
                            dot_row[inner_lvl - lnum] = '...'
                            idx_values[ins_row] = tuple(dot_row)
                level_lengths[lnum] = rec_new
            level_lengths[inner_lvl][ins_row] = 1
            for ix_col in fmt_values:
                fmt_values[ix_col].insert(ins_row, '...')
            nrows += 1
        for i in range(nrows):
            row = []
            tags = {}
            sparse_offset = 0
            j = 0
            for records, v in zip(level_lengths, idx_values[i]):
                if i in records:
                    if records[i] > 1:
                        tags[j] = template.format(span=records[i])
                else:
                    sparse_offset += 1
                    continue
                j += 1
                row.append(v)
            row.extend((fmt_values[j][i] for j in range(self.ncols)))
            if is_truncated_horizontally:
                row.insert(self.row_levels - sparse_offset + self.fmt.tr_col_num, '...')
            self.write_tr(row, indent, self.indent_delta, tags=tags, nindex_levels=len(levels) - sparse_offset)
    else:
        row = []
        for i in range(len(frame)):
            if is_truncated_vertically and i == self.fmt.tr_row_num:
                str_sep_row = ['...'] * len(row)
                self.write_tr(str_sep_row, indent, self.indent_delta, tags=None, nindex_levels=self.row_levels)
            idx_values = list(zip(*frame.index._format_multi(sparsify=False, include_names=False)))
            row = []
            row.extend(idx_values[i])
            row.extend((fmt_values[j][i] for j in range(self.ncols)))
            if is_truncated_horizontally:
                row.insert(self.row_levels + self.fmt.tr_col_num, '...')
            self.write_tr(row, indent, self.indent_delta, tags=None, nindex_levels=frame.index.nlevels)