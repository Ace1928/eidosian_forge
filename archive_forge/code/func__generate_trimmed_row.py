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
def _generate_trimmed_row(self, max_cols: int) -> list:
    """
        When a render has too many rows we generate a trimming row containing "..."

        Parameters
        ----------
        max_cols : int
            Number of permissible columns

        Returns
        -------
        list of elements
        """
    index_headers = [_element('th', f'{self.css['row_heading']} {self.css['level']}{c} {self.css['row_trim']}', '...', not self.hide_index_[c], attributes='') for c in range(self.data.index.nlevels)]
    data: list = []
    visible_col_count: int = 0
    for c, _ in enumerate(self.columns):
        data_element_visible = c not in self.hidden_columns
        if data_element_visible:
            visible_col_count += 1
        if self._check_trim(visible_col_count, max_cols, data, 'td', f'{self.css['data']} {self.css['row_trim']} {self.css['col_trim']}'):
            break
        data.append(_element('td', f'{self.css['data']} {self.css['col']}{c} {self.css['row_trim']}', '...', data_element_visible, attributes=''))
    return index_headers + data