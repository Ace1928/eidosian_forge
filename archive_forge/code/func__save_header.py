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
def _save_header(self) -> None:
    if not self.has_mi_columns or self._has_aliases:
        self.writer.writerow(self.encoded_labels)
    else:
        for row in self._generate_multiindex_header_rows():
            self.writer.writerow(row)