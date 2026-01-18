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
def _update_df(self):
    self._ignore_df_changed = True
    self._df = self.df.copy()
    self._df.insert(0, self._index_col_name, range(0, len(self._df)))
    self._unfiltered_df = self._df.copy()
    self._update_table(update_columns=True, fire_data_change_event=False)
    self._ignore_df_changed = False