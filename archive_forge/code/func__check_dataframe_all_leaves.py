import plotly.graph_objs as go
import plotly.io as pio
from collections import namedtuple, OrderedDict
from ._special_inputs import IdentityMap, Constant, Range
from .trendline_functions import ols, lowess, rolling, expanding, ewm
from _plotly_utils.basevalidators import ColorscaleValidator
from plotly.colors import qualitative, sequential
import math
from packaging import version
import pandas as pd
import numpy as np
from plotly._subplots import (
def _check_dataframe_all_leaves(df):
    df_sorted = df.sort_values(by=list(df.columns))
    null_mask = df_sorted.isnull()
    df_sorted = df_sorted.astype(str)
    null_indices = np.nonzero(null_mask.any(axis=1).values)[0]
    for null_row_index in null_indices:
        row = null_mask.iloc[null_row_index]
        i = np.nonzero(row.values)[0][0]
        if not row[i:].all():
            raise ValueError('None entries cannot have not-None children', df_sorted.iloc[null_row_index])
    df_sorted[null_mask] = ''
    row_strings = list(df_sorted.apply(lambda x: ''.join(x), axis=1))
    for i, row in enumerate(row_strings[:-1]):
        if row_strings[i + 1] in row and i + 1 in null_indices:
            raise ValueError('Non-leaves rows are not permitted in the dataframe \n', df_sorted.iloc[i + 1], 'is not a leaf.')