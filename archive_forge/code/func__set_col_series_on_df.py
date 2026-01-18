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
def _set_col_series_on_df(self, col_name, df, col_series):
    if col_name in self._primary_key:
        if len(self._primary_key) > 1:
            key_index = self._primary_key.index(col_name)
            prev_name = df.index.levels[key_index].name
            df.index.set_levels(col_series, level=key_index, inplace=True)
            df.index.rename(prev_name, level=key_index, inplace=True)
        else:
            prev_name = df.index.name
            df.set_index(col_series, inplace=True)
            df.index.rename(prev_name)
    else:
        df[col_name] = col_series