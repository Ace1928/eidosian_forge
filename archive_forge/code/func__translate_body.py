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
def _translate_body(self, idx_lengths: dict, max_rows: int, max_cols: int):
    """
        Build each <tr> within table <body> as a list

        Use the following structure:
          +--------------------------------------------+---------------------------+
          |  index_header_0    ...    index_header_n   |  data_by_column   ...     |
          +--------------------------------------------+---------------------------+

        Also add elements to the cellstyle_map for more efficient grouped elements in
        <style></style> block

        Parameters
        ----------
        sparsify_index : bool
            Whether index_headers section will add rowspan attributes (>1) to elements.

        Returns
        -------
        body : list
            The associated HTML elements needed for template rendering.
        """
    rlabels = self.data.index.tolist()
    if not isinstance(self.data.index, MultiIndex):
        rlabels = [[x] for x in rlabels]
    body: list = []
    visible_row_count: int = 0
    for r, row_tup in [z for z in enumerate(self.data.itertuples()) if z[0] not in self.hidden_rows]:
        visible_row_count += 1
        if self._check_trim(visible_row_count, max_rows, body, 'row'):
            break
        body_row = self._generate_body_row((r, row_tup, rlabels), max_cols, idx_lengths)
        body.append(body_row)
    return body