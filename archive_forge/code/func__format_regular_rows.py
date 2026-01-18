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
def _format_regular_rows(self) -> Iterable[ExcelCell]:
    if self._has_aliases or self.header:
        self.rowcounter += 1
    if self.index:
        if self.index_label and isinstance(self.index_label, (list, tuple, np.ndarray, Index)):
            index_label = self.index_label[0]
        elif self.index_label and isinstance(self.index_label, str):
            index_label = self.index_label
        else:
            index_label = self.df.index.names[0]
        if isinstance(self.columns, MultiIndex):
            self.rowcounter += 1
        if index_label and self.header is not False:
            yield ExcelCell(self.rowcounter - 1, 0, index_label, self.header_style)
        index_values = self.df.index
        if isinstance(self.df.index, PeriodIndex):
            index_values = self.df.index.to_timestamp()
        for idx, idxval in enumerate(index_values):
            yield CssExcelCell(row=self.rowcounter + idx, col=0, val=idxval, style=self.header_style, css_styles=getattr(self.styler, 'ctx_index', None), css_row=idx, css_col=0, css_converter=self.style_converter)
        coloffset = 1
    else:
        coloffset = 0
    yield from self._generate_body(coloffset)