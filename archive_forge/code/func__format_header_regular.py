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
def _format_header_regular(self) -> Iterable[ExcelCell]:
    if self._has_aliases or self.header:
        coloffset = 0
        if self.index:
            coloffset = 1
            if isinstance(self.df.index, MultiIndex):
                coloffset = len(self.df.index.names)
        colnames = self.columns
        if self._has_aliases:
            self.header = cast(Sequence, self.header)
            if len(self.header) != len(self.columns):
                raise ValueError(f'Writing {len(self.columns)} cols but got {len(self.header)} aliases')
            colnames = self.header
        for colindex, colname in enumerate(colnames):
            yield CssExcelCell(row=self.rowcounter, col=colindex + coloffset, val=colname, style=self.header_style, css_styles=getattr(self.styler, 'ctx_columns', None), css_row=0, css_col=colindex, css_converter=self.style_converter)