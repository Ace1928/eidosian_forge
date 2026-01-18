import ipywidgets as widgets
import pandas as pd
import numpy as np
import json
from types import FunctionType
from IPython.display import display
from numbers import Integral
from traitlets import (
from itertools import chain
from uuid import uuid4
from six import string_types
from distutils.version import LooseVersion
def _handle_change_filter(self, content):
    col_name = content['field']
    columns = self._columns.copy()
    col_info = columns[col_name]
    col_info['filter_info'] = content['filter_info']
    columns[col_name] = col_info
    conditions = []
    for key, value in columns.items():
        if 'filter_info' in value:
            self._append_condition_for_column(key, value['filter_info'], conditions)
    self._columns = columns
    self._ignore_df_changed = True
    if len(conditions) == 0:
        self._df = self._unfiltered_df.copy()
    else:
        combined_condition = conditions[0]
        for c in conditions[1:]:
            combined_condition = combined_condition & c
        self._df = self._unfiltered_df[combined_condition].copy()
    if len(self._df) < self._viewport_range[0]:
        viewport_size = self._viewport_range[1] - self._viewport_range[0]
        range_top = max(0, len(self._df) - viewport_size)
        self._viewport_range = (range_top, range_top + viewport_size)
    self._sorted_column_cache = {}
    self._update_sort()
    self._update_table(triggered_by='change_filter')
    self._ignore_df_changed = False