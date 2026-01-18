import os
import numpy as np
import pandas.json as json
from pandas.tslib import iNaT
from pandas.compat import StringIO, long, u
from pandas import compat, isnull
from pandas import Series, DataFrame, to_datetime, MultiIndex
from pandas.io.common import (get_filepath_or_buffer, _get_handle,
from pandas.core.common import AbstractMethodError
from pandas.formats.printing import pprint_thing
from .normalize import _convert_to_line_delimits
from .table_schema import build_table_schema
def _try_convert_types(self):
    if self.obj is None:
        return
    if self.convert_dates:
        self._try_convert_dates()
    self._process_converter(lambda col, c: self._try_convert_data(col, c, convert_dates=False))