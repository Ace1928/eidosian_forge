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
class PolyEdit(PolyDraw):
    """
    Attaches a PolyEditTool and syncs the datasource.

    shared: boolean
        Whether PolyEditTools should be shared between multiple elements

    tooltip: str
        An optional tooltip to override the default

    vertex_style: dict
        A dictionary specifying the style options for the vertices.
        The usual bokeh style options apply, e.g. fill_color,
        line_alpha, size, etc.
    """
    data = param.Dict(constant=True, doc='\n        Data synced from Bokeh ColumnDataSource supplied as a\n        dictionary of columns, where each column is a list of values\n        (for point-like data) or list of lists of values (for\n        path-like data).')

    def __init__(self, vertex_style=None, shared=True, **params):
        if vertex_style is None:
            vertex_style = {}
        self.shared = shared
        super().__init__(vertex_style=vertex_style, **params)