import operator
import sys
from types import BuiltinFunctionType, BuiltinMethodType, FunctionType, MethodType
import numpy as np
import pandas as pd
import param
from ..core.data import PandasInterface
from ..core.dimension import Dimension
from ..core.util import flatten, resolve_dependent_value, unique_iterator
def _maybe_map(numpy_fn):

    def fn(values, *args, **kwargs):
        series_like = hasattr(values, 'index') and (not isinstance(values, list))
        map_fn = getattr(values, 'map_partitions', None) or getattr(values, 'map_blocks', None)
        if map_fn:
            if series_like:
                return map_fn(lambda s: type(s)(numpy_fn(s, *args, **kwargs), index=s.index))
            else:
                return map_fn(lambda s: numpy_fn(s, *args, **kwargs))
        elif series_like:
            return type(values)(numpy_fn(values, *args, **kwargs), index=values.index)
        else:
            return numpy_fn(values, *args, **kwargs)
    return fn