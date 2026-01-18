import operator
import sys
from types import BuiltinFunctionType, BuiltinMethodType, FunctionType, MethodType
import numpy as np
import pandas as pd
import param
from ..core.data import PandasInterface
from ..core.dimension import Dimension
from ..core.util import flatten, resolve_dependent_value, unique_iterator
class loc:
    """Implements loc for dim expressions.
    """
    __name__ = 'loc'

    def __init__(self, dim_expr):
        self.expr = dim_expr
        self.index = slice(None)

    def __getitem__(self, index):
        self.index = index
        return dim(self.expr, self)

    def __call__(self, values):
        return values.loc[resolve_dependent_value(self.index)]