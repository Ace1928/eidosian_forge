import sys
from collections.abc import Hashable
from functools import wraps
from packaging.version import Version
from types import FunctionType
import bokeh
import numpy as np
import pandas as pd
import param
import holoviews as hv
def _convert_col_names_to_str(data):
    """
    Convert column names to string.
    """
    if not hasattr(data, 'columns') or not hasattr(data, 'rename'):
        return data
    renamed = {c: str(c) for c in data.columns if not isinstance(c, str) and isinstance(c, Hashable)}
    if renamed:
        data = data.rename(columns=renamed)
    return data