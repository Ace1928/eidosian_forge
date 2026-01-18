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
def _process_columnattributes_subheader(self, offset: int, length: int) -> None:
    int_len = self._int_length
    column_attributes_vectors_count = (length - 2 * int_len - 12) // (int_len + 8)
    for i in range(column_attributes_vectors_count):
        col_data_offset = offset + int_len + const.column_data_offset_offset + i * (int_len + 8)
        col_data_len = offset + 2 * int_len + const.column_data_length_offset + i * (int_len + 8)
        col_types = offset + 2 * int_len + const.column_type_offset + i * (int_len + 8)
        x = self._read_uint(col_data_offset, int_len)
        self._column_data_offsets.append(x)
        x = self._read_uint(col_data_len, const.column_data_length_length)
        self._column_data_lengths.append(x)
        x = self._read_uint(col_types, const.column_type_length)
        self._column_types.append(b'd' if x == 1 else b's')