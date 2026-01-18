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
class SeriesWriter(Writer):
    _default_orient = 'index'

    def _format_axes(self):
        if not self.obj.index.is_unique and self.orient == 'index':
            raise ValueError("Series index must be unique for orient='%s'" % self.orient)