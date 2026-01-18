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
def _chunk_to_dataframe(self) -> DataFrame:
    n = self._current_row_in_chunk_index
    m = self._current_row_in_file_index
    ix = range(m - n, m)
    rslt = {}
    js, jb = (0, 0)
    for j in range(self.column_count):
        name = self.column_names[j]
        if self._column_types[j] == b'd':
            col_arr = self._byte_chunk[jb, :].view(dtype=self.byte_order + 'd')
            rslt[name] = pd.Series(col_arr, dtype=np.float64, index=ix, copy=False)
            if self.convert_dates:
                if self.column_formats[j] in const.sas_date_formats:
                    rslt[name] = _convert_datetimes(rslt[name], 'd')
                elif self.column_formats[j] in const.sas_datetime_formats:
                    rslt[name] = _convert_datetimes(rslt[name], 's')
            jb += 1
        elif self._column_types[j] == b's':
            rslt[name] = pd.Series(self._string_chunk[js, :], index=ix, copy=False)
            if self.convert_text and self.encoding is not None:
                rslt[name] = self._decode_string(rslt[name].str)
            js += 1
        else:
            self.close()
            raise ValueError(f'unknown column type {repr(self._column_types[j])}')
    df = DataFrame(rslt, columns=self.column_names, index=ix, copy=False)
    return df