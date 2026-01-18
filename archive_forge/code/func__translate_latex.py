from __future__ import annotations
from collections import defaultdict
from collections.abc import Sequence
from functools import partial
import re
from typing import (
from uuid import uuid4
import numpy as np
from pandas._config import get_option
from pandas._libs import lib
from pandas.compat._optional import import_optional_dependency
from pandas.core.dtypes.common import (
from pandas.core.dtypes.generic import ABCSeries
from pandas import (
from pandas.api.types import is_list_like
import pandas.core.common as com
from markupsafe import escape as escape_html  # markupsafe is jinja2 dependency
def _translate_latex(self, d: dict, clines: str | None) -> None:
    """
        Post-process the default render dict for the LaTeX template format.

        Processing items included are:
          - Remove hidden columns from the non-headers part of the body.
          - Place cellstyles directly in td cells rather than use cellstyle_map.
          - Remove hidden indexes or reinsert missing th elements if part of multiindex
            or multirow sparsification (so that \\multirow and \\multicol work correctly).
        """
    index_levels = self.index.nlevels
    visible_index_level_n = index_levels - sum(self.hide_index_)
    d['head'] = [[{**col, 'cellstyle': self.ctx_columns[r, c - visible_index_level_n]} for c, col in enumerate(row) if col['is_visible']] for r, row in enumerate(d['head'])]

    def _concatenated_visible_rows(obj, n, row_indices):
        """
            Extract all visible row indices recursively from concatenated stylers.
            """
        row_indices.extend([r + n for r in range(len(obj.index)) if r not in obj.hidden_rows])
        n += len(obj.index)
        for concatenated in obj.concatenated:
            n = _concatenated_visible_rows(concatenated, n, row_indices)
        return n

    def concatenated_visible_rows(obj):
        row_indices: list[int] = []
        _concatenated_visible_rows(obj, 0, row_indices)
        return row_indices
    body = []
    for r, row in zip(concatenated_visible_rows(self), d['body']):
        if all(self.hide_index_):
            row_body_headers = []
        else:
            row_body_headers = [{**col, 'display_value': col['display_value'] if col['is_visible'] else '', 'cellstyle': self.ctx_index[r, c]} for c, col in enumerate(row[:index_levels]) if col['type'] == 'th' and (not self.hide_index_[c])]
        row_body_cells = [{**col, 'cellstyle': self.ctx[r, c]} for c, col in enumerate(row[index_levels:]) if col['is_visible'] and col['type'] == 'td']
        body.append(row_body_headers + row_body_cells)
    d['body'] = body
    if clines not in [None, 'all;data', 'all;index', 'skip-last;data', 'skip-last;index']:
        raise ValueError(f"`clines` value of {clines} is invalid. Should either be None or one of 'all;data', 'all;index', 'skip-last;data', 'skip-last;index'.")
    if clines is not None:
        data_len = len(row_body_cells) if 'data' in clines and d['body'] else 0
        d['clines'] = defaultdict(list)
        visible_row_indexes: list[int] = [r for r in range(len(self.data.index)) if r not in self.hidden_rows]
        visible_index_levels: list[int] = [i for i in range(index_levels) if not self.hide_index_[i]]
        for rn, r in enumerate(visible_row_indexes):
            for lvln, lvl in enumerate(visible_index_levels):
                if lvl == index_levels - 1 and 'skip-last' in clines:
                    continue
                idx_len = d['index_lengths'].get((lvl, r), None)
                if idx_len is not None:
                    d['clines'][rn + idx_len].append(f'\\cline{{{lvln + 1}-{len(visible_index_levels) + data_len}}}')