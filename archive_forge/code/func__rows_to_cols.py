from __future__ import annotations
from collections import (
from collections.abc import (
import csv
from io import StringIO
import re
from typing import (
import warnings
import numpy as np
from pandas._libs import lib
from pandas.errors import (
from pandas.util._decorators import cache_readonly
from pandas.util._exceptions import find_stack_level
from pandas.core.dtypes.common import (
from pandas.core.dtypes.inference import is_dict_like
from pandas.io.common import (
from pandas.io.parsers.base_parser import (
def _rows_to_cols(self, content: list[list[Scalar]]) -> list[np.ndarray]:
    col_len = self.num_original_columns
    if self._implicit_index:
        col_len += len(self.index_col)
    max_len = max((len(row) for row in content))
    if max_len > col_len and self.index_col is not False and (self.usecols is None):
        footers = self.skipfooter if self.skipfooter else 0
        bad_lines = []
        iter_content = enumerate(content)
        content_len = len(content)
        content = []
        for i, _content in iter_content:
            actual_len = len(_content)
            if actual_len > col_len:
                if callable(self.on_bad_lines):
                    new_l = self.on_bad_lines(_content)
                    if new_l is not None:
                        content.append(new_l)
                elif self.on_bad_lines in (self.BadLineHandleMethod.ERROR, self.BadLineHandleMethod.WARN):
                    row_num = self.pos - (content_len - i + footers)
                    bad_lines.append((row_num, actual_len))
                    if self.on_bad_lines == self.BadLineHandleMethod.ERROR:
                        break
            else:
                content.append(_content)
        for row_num, actual_len in bad_lines:
            msg = f'Expected {col_len} fields in line {row_num + 1}, saw {actual_len}'
            if self.delimiter and len(self.delimiter) > 1 and (self.quoting != csv.QUOTE_NONE):
                reason = 'Error could possibly be due to quotes being ignored when a multi-char delimiter is used.'
                msg += '. ' + reason
            self._alert_malformed(msg, row_num + 1)
    zipped_content = list(lib.to_object_array(content, min_width=col_len).T)
    if self.usecols:
        assert self._col_indices is not None
        col_indices = self._col_indices
        if self._implicit_index:
            zipped_content = [a for i, a in enumerate(zipped_content) if i < len(self.index_col) or i - len(self.index_col) in col_indices]
        else:
            zipped_content = [a for i, a in enumerate(zipped_content) if i in col_indices]
    return zipped_content