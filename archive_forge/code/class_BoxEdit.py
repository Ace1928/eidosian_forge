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
class BoxEdit(CDSStream):
    """
    Attaches a BoxEditTool and syncs the datasource.

    empty_value: int/float/string/None
        The value to insert on non-position columns when adding a new box

    num_objects: int
        The number of boxes that can be drawn before overwriting the
        oldest drawn box.

    styles: dict
        A dictionary specifying lists of styles to cycle over whenever
        a new box glyph is drawn.

    tooltip: str
        An optional tooltip to override the default
    """
    data = param.Dict(constant=True, doc='\n        Data synced from Bokeh ColumnDataSource supplied as a\n        dictionary of columns, where each column is a list of values\n        (for point-like data) or list of lists of values (for\n        path-like data).')

    def __init__(self, empty_value=None, num_objects=0, styles=None, tooltip=None, **params):
        if styles is None:
            styles = {}
        self.empty_value = empty_value
        self.num_objects = num_objects
        self.styles = styles
        self.tooltip = tooltip
        super().__init__(**params)

    @property
    def element(self):
        from .element import Polygons, Rectangles
        source = self.source
        if isinstance(source, UniformNdMapping):
            source = source.last
        data = self.data
        if not data:
            return source.clone([])
        dims = ['x0', 'y0', 'x1', 'y1'] + [vd.name for vd in source.vdims]
        if isinstance(source, Rectangles):
            data = tuple((data[d] for d in dims))
            return source.clone(data, id=None)
        paths = []
        for i, (x0, x1, y0, y1) in enumerate(zip(data['x0'], data['x1'], data['y0'], data['y1'])):
            xs = [x0, x0, x1, x1]
            ys = [y0, y1, y1, y0]
            if isinstance(source, Polygons):
                xs.append(x0)
                ys.append(y0)
            vals = [data[vd.name][i] for vd in source.vdims]
            paths.append((xs, ys) + tuple(vals))
        datatype = source.datatype if source.interface.multi else ['multitabular']
        return source.clone(paths, datatype=datatype, id=None)

    @property
    def dynamic(self):
        from .core.spaces import DynamicMap
        return DynamicMap(lambda *args, **kwargs: self.element, streams=[self])