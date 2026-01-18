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
def _initialize_sort_column(self, col_name, to_timestamp=False):
    sort_column_name = self._sort_helper_columns.get(col_name)
    if sort_column_name:
        return sort_column_name
    sort_col_series = self._get_col_series_from_df(col_name, self._df)
    sort_col_series_unfiltered = self._get_col_series_from_df(col_name, self._unfiltered_df)
    sort_column_name = str(col_name) + self._sort_col_suffix
    if to_timestamp:
        self._df[sort_column_name] = sort_col_series.to_timestamp()
        self._unfiltered_df[sort_column_name] = sort_col_series_unfiltered.to_timestamp()
    else:
        self._df[sort_column_name] = sort_col_series.map(str)
        self._unfiltered_df[sort_column_name] = sort_col_series_unfiltered.map(str)
    self._sort_helper_columns[col_name] = sort_column_name
    return sort_column_name