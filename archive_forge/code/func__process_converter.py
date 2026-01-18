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
def _process_converter(self, f, filt=lambda col, c: True):
    """ take a conversion function and possibly recreate the frame """
    needs_new_obj = False
    new_obj = dict()
    for i, (col, c) in enumerate(self.obj.iteritems()):
        if filt(col, c):
            new_data, result = f(col, c)
            if result:
                c = new_data
                needs_new_obj = True
        new_obj[i] = c
    if needs_new_obj:
        new_obj = DataFrame(new_obj, index=self.obj.index)
        new_obj.columns = self.obj.columns
        self.obj = new_obj