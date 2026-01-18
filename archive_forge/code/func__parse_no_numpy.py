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
def _parse_no_numpy(self):
    json = self.json
    orient = self.orient
    if orient == 'columns':
        self.obj = DataFrame(loads(json, precise_float=self.precise_float), dtype=None)
    elif orient == 'split':
        decoded = dict(((str(k), v) for k, v in compat.iteritems(loads(json, precise_float=self.precise_float))))
        self.check_keys_split(decoded)
        self.obj = DataFrame(dtype=None, **decoded)
    elif orient == 'index':
        self.obj = DataFrame(loads(json, precise_float=self.precise_float), dtype=None).T
    else:
        self.obj = DataFrame(loads(json, precise_float=self.precise_float), dtype=None)