from __future__ import annotations
from abc import (
from typing import (
import numpy as np
from pandas.core.dtypes.generic import ABCMultiIndex
class RowHeaderIterator(RowStringIterator):
    """Iterator for the table header rows."""

    def __iter__(self) -> Iterator[str]:
        for row_num in range(len(self.strrows)):
            if row_num < self._header_row_num:
                yield self.get_strrow(row_num)