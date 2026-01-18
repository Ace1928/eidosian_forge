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
def _try_convert_dates(self):
    if self.obj is None:
        return
    convert_dates = self.convert_dates
    if convert_dates is True:
        convert_dates = []
    convert_dates = set(convert_dates)

    def is_ok(col):
        """ return if this col is ok to try for a date parse """
        if not isinstance(col, compat.string_types):
            return False
        col_lower = col.lower()
        if col_lower.endswith('_at') or col_lower.endswith('_time') or col_lower == 'modified' or (col_lower == 'date') or (col_lower == 'datetime') or col_lower.startswith('timestamp'):
            return True
        return False
    self._process_converter(lambda col, c: self._try_convert_to_date(c), lambda col, c: self.keep_default_dates and is_ok(col) or col in convert_dates)