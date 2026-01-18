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
class FrameParser(Parser):
    _default_orient = 'columns'
    _split_keys = ('columns', 'index', 'data')

    def _parse_numpy(self):
        json = self.json
        orient = self.orient
        if orient == 'columns':
            args = loads(json, dtype=None, numpy=True, labelled=True, precise_float=self.precise_float)
            if args:
                args = (args[0].T, args[2], args[1])
            self.obj = DataFrame(*args)
        elif orient == 'split':
            decoded = loads(json, dtype=None, numpy=True, precise_float=self.precise_float)
            decoded = dict(((str(k), v) for k, v in compat.iteritems(decoded)))
            self.check_keys_split(decoded)
            self.obj = DataFrame(**decoded)
        elif orient == 'values':
            self.obj = DataFrame(loads(json, dtype=None, numpy=True, precise_float=self.precise_float))
        else:
            self.obj = DataFrame(*loads(json, dtype=None, numpy=True, labelled=True, precise_float=self.precise_float))

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

    def _try_convert_types(self):
        if self.obj is None:
            return
        if self.convert_dates:
            self._try_convert_dates()
        self._process_converter(lambda col, c: self._try_convert_data(col, c, convert_dates=False))

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