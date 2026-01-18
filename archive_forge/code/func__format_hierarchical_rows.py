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
def _format_hierarchical_rows(self) -> Iterable[ExcelCell]:
    if self._has_aliases or self.header:
        self.rowcounter += 1
    gcolidx = 0
    if self.index:
        index_labels = self.df.index.names
        if self.index_label and isinstance(self.index_label, (list, tuple, np.ndarray, Index)):
            index_labels = self.index_label
        if isinstance(self.columns, MultiIndex) and self.merge_cells:
            self.rowcounter += 1
        if com.any_not_none(*index_labels) and self.header is not False:
            for cidx, name in enumerate(index_labels):
                yield ExcelCell(self.rowcounter - 1, cidx, name, self.header_style)
        if self.merge_cells:
            level_strs = self.df.index._format_multi(sparsify=True, include_names=False)
            level_lengths = get_level_lengths(level_strs)
            for spans, levels, level_codes in zip(level_lengths, self.df.index.levels, self.df.index.codes):
                values = levels.take(level_codes, allow_fill=levels._can_hold_na, fill_value=levels._na_value)
                for i, span_val in spans.items():
                    mergestart, mergeend = (None, None)
                    if span_val > 1:
                        mergestart = self.rowcounter + i + span_val - 1
                        mergeend = gcolidx
                    yield CssExcelCell(row=self.rowcounter + i, col=gcolidx, val=values[i], style=self.header_style, css_styles=getattr(self.styler, 'ctx_index', None), css_row=i, css_col=gcolidx, css_converter=self.style_converter, mergestart=mergestart, mergeend=mergeend)
                gcolidx += 1
        else:
            for indexcolvals in zip(*self.df.index):
                for idx, indexcolval in enumerate(indexcolvals):
                    yield CssExcelCell(row=self.rowcounter + idx, col=gcolidx, val=indexcolval, style=self.header_style, css_styles=getattr(self.styler, 'ctx_index', None), css_row=idx, css_col=gcolidx, css_converter=self.style_converter)
                gcolidx += 1
    yield from self._generate_body(gcolidx)