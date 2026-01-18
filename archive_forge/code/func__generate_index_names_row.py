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
def _generate_index_names_row(self, iter: Sequence, max_cols: int, col_lengths: dict):
    """
        Generate the row containing index names

         +----------------------------+---------------+---------------------------+
         |  index_names (level_0 to level_n) ...      | column_blanks ...         |
         +----------------------------+---------------+---------------------------+

        Parameters
        ----------
        iter : tuple
            Looping variables from outer scope
        max_cols : int
            Permissible number of columns

        Returns
        -------
        list of elements
        """
    clabels = iter
    index_names = [_element('th', f'{self.css['index_name']} {self.css['level']}{c}', self.css['blank_value'] if name is None else name, not self.hide_index_[c]) for c, name in enumerate(self.data.index.names)]
    column_blanks: list = []
    visible_col_count: int = 0
    if clabels:
        last_level = self.columns.nlevels - 1
        for c, value in enumerate(clabels[last_level]):
            header_element_visible = _is_visible(c, last_level, col_lengths)
            if header_element_visible:
                visible_col_count += 1
            if self._check_trim(visible_col_count, max_cols, column_blanks, 'th', f'{self.css['blank']} {self.css['col']}{c} {self.css['col_trim']}', self.css['blank_value']):
                break
            column_blanks.append(_element('th', f'{self.css['blank']} {self.css['col']}{c}', self.css['blank_value'], c not in self.hidden_columns))
    return index_names + column_blanks