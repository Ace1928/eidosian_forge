import sys
import weakref
from collections import defaultdict
from contextlib import contextmanager
from functools import partial
from itertools import groupby
from numbers import Number
from types import FunctionType
import numpy as np
import pandas as pd
import param
from packaging.version import Version
from .core import util
from .core.ndmapping import UniformNdMapping
class CurveEdit(PointDraw):
    """
    Attaches a PointDraw to the plot which allows editing the Curve when selected.

    style: dict
        A dictionary specifying the style of the vertices.

    tooltip: str
        An optional tooltip to override the default
    """
    data = param.Dict(constant=True, doc='\n        Data synced from Bokeh ColumnDataSource supplied as a\n        dictionary of columns, where each column is a list of values\n        (for point-like data) or list of lists of values (for\n        path-like data).')

    def __init__(self, style=None, tooltip=None, **params):
        if style is None:
            style = {}
        self.style = style or {'size': 10}
        self.tooltip = tooltip
        super(PointDraw, self).__init__(**params)