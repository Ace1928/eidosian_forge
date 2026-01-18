from __future__ import annotations
from collections.abc import (
import functools
import itertools
import re
from typing import (
import warnings
import numpy as np
from pandas._libs.lib import is_list_like
from pandas.util._decorators import doc
from pandas.util._exceptions import find_stack_level
from pandas.core.dtypes import missing
from pandas.core.dtypes.common import (
from pandas import (
import pandas.core.common as com
from pandas.core.shared_docs import _shared_docs
from pandas.io.formats._color_data import CSS4_COLORS
from pandas.io.formats.css import (
from pandas.io.formats.format import get_level_lengths
from pandas.io.formats.printing import pprint_thing
def _format_header_mi(self) -> Iterable[ExcelCell]:
    if self.columns.nlevels > 1:
        if not self.index:
            raise NotImplementedError("Writing to Excel with MultiIndex columns and no index ('index'=False) is not yet implemented.")
    if not (self._has_aliases or self.header):
        return
    columns = self.columns
    level_strs = columns._format_multi(sparsify=self.merge_cells, include_names=False)
    level_lengths = get_level_lengths(level_strs)
    coloffset = 0
    lnum = 0
    if self.index and isinstance(self.df.index, MultiIndex):
        coloffset = len(self.df.index[0]) - 1
    if self.merge_cells:
        for lnum, name in enumerate(columns.names):
            yield ExcelCell(row=lnum, col=coloffset, val=name, style=self.header_style)
        for lnum, (spans, levels, level_codes) in enumerate(zip(level_lengths, columns.levels, columns.codes)):
            values = levels.take(level_codes)
            for i, span_val in spans.items():
                mergestart, mergeend = (None, None)
                if span_val > 1:
                    mergestart, mergeend = (lnum, coloffset + i + span_val)
                yield CssExcelCell(row=lnum, col=coloffset + i + 1, val=values[i], style=self.header_style, css_styles=getattr(self.styler, 'ctx_columns', None), css_row=lnum, css_col=i, css_converter=self.style_converter, mergestart=mergestart, mergeend=mergeend)
    else:
        for i, values in enumerate(zip(*level_strs)):
            v = '.'.join(map(pprint_thing, values))
            yield CssExcelCell(row=lnum, col=coloffset + i + 1, val=v, style=self.header_style, css_styles=getattr(self.styler, 'ctx_columns', None), css_row=lnum, css_col=i, css_converter=self.style_converter)
    self.rowcounter = lnum