import operator
import sys
from types import BuiltinFunctionType, BuiltinMethodType, FunctionType, MethodType
import numpy as np
import pandas as pd
import param
from ..core.data import PandasInterface
from ..core.dimension import Dimension
from ..core.util import flatten, resolve_dependent_value, unique_iterator
@property
def _current_accessor(self):
    if self.ops and self.ops[-1]['kwargs'].get('accessor'):
        return self.ops[-1]['fn']