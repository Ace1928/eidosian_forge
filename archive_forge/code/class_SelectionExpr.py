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
class SelectionExpr(Derived):
    selection_expr = param.Parameter(default=None, constant=True)
    bbox = param.Dict(default=None, constant=True)
    region_element = param.Parameter(default=None, constant=True)

    def __init__(self, source, include_region=True, **params):
        from .core.spaces import DynamicMap
        from .element import Element
        from .plotting.util import initialize_dynamic
        self._index_cols = params.pop('index_cols', None)
        self.include_region = include_region
        if isinstance(source, DynamicMap):
            initialize_dynamic(source)
        if not (isinstance(source, DynamicMap) and issubclass(source.type, Element) or isinstance(source, Element)):
            raise ValueError(f'The source of SelectionExpr must be an instance of an Element subclass or a DynamicMap that returns such an instance. Received value of type {type(source)}: {source}')
        input_streams = self._build_selection_streams(source)
        super().__init__(source=source, input_streams=input_streams, exclusive=True, **params)

    def clone(self):
        return type(self)(self.source, **self.contents)

    def _build_selection_streams(self, source):
        from holoviews.core.spaces import DynamicMap
        if isinstance(source, DynamicMap):
            element_type = source.type
        else:
            element_type = source
        if element_type:
            input_streams = []
            for stream in element_type._selection_streams:
                kwargs = dict(source=source)
                if isinstance(stream, Selection1D):
                    kwargs['index'] = None
                input_streams.append(stream(**kwargs))
            return input_streams
        else:
            return []

    @property
    def constants(self):
        return {'source': self.source, 'index_cols': self._index_cols, 'include_region': self.include_region}

    def transform(self):
        for stream in self.input_streams:
            if isinstance(stream, Selection1D) and stream._triggering and (not self._index_cols):
                return
        return super().transform()

    @classmethod
    def transform_function(cls, stream_values, constants):
        hvobj = constants['source']
        include_region = constants['include_region']
        if hvobj is None:
            return dict(selection_expr=None, bbox=None, region_element=None)
        from holoviews.core.spaces import DynamicMap
        if isinstance(hvobj, DynamicMap):
            element = hvobj.values()[-1]
        else:
            element = hvobj
        selection_expr = None
        bbox = None
        region_element = None
        for stream_value in stream_values:
            params = dict(stream_value, index_cols=constants['index_cols'])
            selection = element._get_selection_expr_for_stream_value(**params)
            if selection is None:
                return
            selection_expr, bbox, region_element = selection
            if selection_expr is not None:
                break
        for expr_transform in element._transforms[::-1]:
            if selection_expr is not None:
                selection_expr = expr_transform(selection_expr)
        return dict(selection_expr=selection_expr, bbox=bbox, region_element=region_element if include_region else None)

    @property
    def source(self):
        return Stream.source.fget(self)

    @source.setter
    def source(self, value):
        self._unregister_input_streams()
        Stream.source.fset(self, value)
        if self.source is not None:
            input_streams = self._build_selection_streams(self.source)
        else:
            input_streams = []
        self.update(selection_expr=None, bbox=None, region_element=None)
        self._register_streams(input_streams)