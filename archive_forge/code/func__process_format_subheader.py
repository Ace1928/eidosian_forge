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
def _process_format_subheader(self, offset: int, length: int) -> None:
    int_len = self._int_length
    text_subheader_format = offset + const.column_format_text_subheader_index_offset + 3 * int_len
    col_format_offset = offset + const.column_format_offset_offset + 3 * int_len
    col_format_len = offset + const.column_format_length_offset + 3 * int_len
    text_subheader_label = offset + const.column_label_text_subheader_index_offset + 3 * int_len
    col_label_offset = offset + const.column_label_offset_offset + 3 * int_len
    col_label_len = offset + const.column_label_length_offset + 3 * int_len
    x = self._read_uint(text_subheader_format, const.column_format_text_subheader_index_length)
    format_idx = min(x, len(self.column_names_raw) - 1)
    format_start = self._read_uint(col_format_offset, const.column_format_offset_length)
    format_len = self._read_uint(col_format_len, const.column_format_length_length)
    label_idx = self._read_uint(text_subheader_label, const.column_label_text_subheader_index_length)
    label_idx = min(label_idx, len(self.column_names_raw) - 1)
    label_start = self._read_uint(col_label_offset, const.column_label_offset_length)
    label_len = self._read_uint(col_label_len, const.column_label_length_length)
    label_names = self.column_names_raw[label_idx]
    column_label = self._convert_header_text(label_names[label_start:label_start + label_len])
    format_names = self.column_names_raw[format_idx]
    column_format = self._convert_header_text(format_names[format_start:format_start + format_len])
    current_column_number = len(self.columns)
    col = _Column(current_column_number, self.column_names[current_column_number], column_label, column_format, self._column_types[current_column_number], self._column_data_lengths[current_column_number])
    self.column_formats.append(column_format)
    self.columns.append(col)