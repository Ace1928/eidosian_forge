import logging
import bisect
from pyomo.core.expr.numvalue import value as _value
from pyomo.core.kernel.set_types import IntegerSet
from pyomo.core.kernel.block import block
from pyomo.core.kernel.expression import expression, expression_tuple
from pyomo.core.kernel.variable import (
from pyomo.core.kernel.constraint import (
from pyomo.core.kernel.sos import sos2
from pyomo.core.kernel.piecewise_library.util import (
class _shadow_list(object):
    __slots__ = ('_x',)

    def __init__(self, x):
        self._x = x

    def __len__(self):
        return self._x.__len__()

    def __getitem__(self, i):
        return _value(self._x.__getitem__(i))