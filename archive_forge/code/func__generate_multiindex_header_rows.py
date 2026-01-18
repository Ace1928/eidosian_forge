from __future__ import annotations
from collections.abc import (
import csv as csvlib
import os
from typing import (
import numpy as np
from pandas._libs import writers as libwriters
from pandas._typing import SequenceNotStr
from pandas.util._decorators import cache_readonly
from pandas.core.dtypes.generic import (
from pandas.core.dtypes.missing import notna
from pandas.core.indexes.api import Index
from pandas.io.common import get_handle
def _generate_multiindex_header_rows(self) -> Iterator[list[Hashable]]:
    columns = self.obj.columns
    for i in range(columns.nlevels):
        col_line = []
        if self.index:
            col_line.append(columns.names[i])
            if isinstance(self.index_label, list) and len(self.index_label) > 1:
                col_line.extend([''] * (len(self.index_label) - 1))
        col_line.extend(columns._get_level_values(i))
        yield col_line
    if self.encoded_labels and set(self.encoded_labels) != {''}:
        yield (self.encoded_labels + [''] * len(columns))