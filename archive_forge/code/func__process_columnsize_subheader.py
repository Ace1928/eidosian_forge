from __future__ import annotations
from collections import abc
from datetime import (
import sys
from typing import TYPE_CHECKING
import numpy as np
from pandas._libs.byteswap import (
from pandas._libs.sas import (
from pandas._libs.tslibs.conversion import cast_from_unit_vectorized
from pandas.errors import EmptyDataError
import pandas as pd
from pandas import (
from pandas.io.common import get_handle
import pandas.io.sas.sas_constants as const
from pandas.io.sas.sasreader import ReaderBase
def _process_columnsize_subheader(self, offset: int, length: int) -> None:
    int_len = self._int_length
    offset += int_len
    self.column_count = self._read_uint(offset, int_len)
    if self.col_count_p1 + self.col_count_p2 != self.column_count:
        print(f'Warning: column count mismatch ({self.col_count_p1} + {self.col_count_p2} != {self.column_count})\n')