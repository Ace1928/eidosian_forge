import copy
import sys
from functools import wraps
from types import FunctionType
import param
from . import util
from .pprint import PrettyPrinter
def _transform_dimension(self, kdims, vdims, dimension):
    if dimension in kdims:
        idx = kdims.index(dimension)
        dimension = self._obj.kdims[idx]
    elif dimension in vdims:
        idx = vdims.index(dimension)
        dimension = self._obj.vdims[idx]
    return dimension