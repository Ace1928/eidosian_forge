import collections
import functools
import inspect
from collections.abc import Sequence
from collections.abc import Mapping
from pyomo.common.dependencies import numpy, numpy_available, pandas, pandas_available
from pyomo.common.modeling import NOTSET
from pyomo.core.pyomoobject import PyomoObject
class ConstantInitializer(InitializerBase):
    """Initializer for constant values"""
    __slots__ = ('val', 'verified')

    def __init__(self, val):
        self.val = val
        self.verified = False

    def __call__(self, parent, idx):
        return self.val

    def constant(self):
        return True