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
def _translate_header(self, sparsify_cols: bool, max_cols: int):
    """
        Build each <tr> within table <head> as a list

        Using the structure:
             +----------------------------+---------------+---------------------------+
             |  index_blanks ...          | column_name_0 |  column_headers (level_0) |
          1) |       ..                   |       ..      |             ..            |
             |  index_blanks ...          | column_name_n |  column_headers (level_n) |
             +----------------------------+---------------+---------------------------+
          2) |  index_names (level_0 to level_n) ...      | column_blanks ...         |
             +----------------------------+---------------+---------------------------+

        Parameters
        ----------
        sparsify_cols : bool
            Whether column_headers section will add colspan attributes (>1) to elements.
        max_cols : int
            Maximum number of columns to render. If exceeded will contain `...` filler.

        Returns
        -------
        head : list
            The associated HTML elements needed for template rendering.
        """
    col_lengths = _get_level_lengths(self.columns, sparsify_cols, max_cols, self.hidden_columns)
    clabels = self.data.columns.tolist()
    if self.data.columns.nlevels == 1:
        clabels = [[x] for x in clabels]
    clabels = list(zip(*clabels))
    head = []
    for r, hide in enumerate(self.hide_columns_):
        if hide or not clabels:
            continue
        header_row = self._generate_col_header_row((r, clabels), max_cols, col_lengths)
        head.append(header_row)
    if self.data.index.names and com.any_not_none(*self.data.index.names) and (not all(self.hide_index_)) and (not self.hide_index_names):
        index_names_row = self._generate_index_names_row(clabels, max_cols, col_lengths)
        head.append(index_names_row)
    return head