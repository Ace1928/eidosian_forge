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
def _append_condition_for_column(self, col_name, filter_info, conditions):
    col_series = self._get_col_series_from_df(col_name, self._unfiltered_df)
    if filter_info['type'] == 'slider':
        if filter_info['min'] is not None:
            conditions.append(col_series >= filter_info['min'])
        if filter_info['max'] is not None:
            conditions.append(col_series <= filter_info['max'])
    elif filter_info['type'] == 'date':
        if filter_info['min'] is not None:
            conditions.append(col_series >= pd.to_datetime(filter_info['min'], unit='ms'))
        if filter_info['max'] is not None:
            conditions.append(col_series <= pd.to_datetime(filter_info['max'], unit='ms'))
    elif filter_info['type'] == 'boolean':
        if filter_info['selected'] is not None:
            conditions.append(col_series == filter_info['selected'])
    elif filter_info['type'] == 'text':
        if col_name not in self._filter_tables:
            return
        col_filter_table = self._filter_tables[col_name]
        selected_indices = filter_info['selected']
        excluded_indices = filter_info['excluded']

        def get_value_from_filter_table(i):
            return col_filter_table[i]
        if selected_indices == 'all':
            if excluded_indices is not None and len(excluded_indices) > 0:
                excluded_values = list(map(get_value_from_filter_table, excluded_indices))
                conditions.append(~col_series.isin(excluded_values))
        elif selected_indices is not None and len(selected_indices) > 0:
            selected_values = list(map(get_value_from_filter_table, selected_indices))
            conditions.append(col_series.isin(selected_values))