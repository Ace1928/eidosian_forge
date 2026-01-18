import inspect
import itertools
import logging
import math
import sys
import weakref
from pyomo.common.pyomo_typing import overload
from pyomo.common.collections import ComponentSet
from pyomo.common.deprecation import deprecated, deprecation_warning, RenamedClass
from pyomo.common.errors import DeveloperError, PyomoException
from pyomo.common.log import is_debug_set
from pyomo.common.modeling import NOTSET
from pyomo.common.sorting import sorted_robust
from pyomo.common.timing import ConstructionTimer
from pyomo.core.expr.numvalue import (
from pyomo.core.base.disable_methods import disable_methods
from pyomo.core.base.initializer import (
from pyomo.core.base.range import (
from pyomo.core.base.component import (
from pyomo.core.base.indexed_component import (
from pyomo.core.base.global_set import (
from collections.abc import Sequence
from operator import itemgetter
class _AnySet(_SetData, Set):

    def __init__(self, **kwds):
        _SetData.__init__(self, component=self)
        kwds.setdefault('domain', self)
        Set.__init__(self, **kwds)
        self.construct()

    def get(self, val, default=None):
        return val if val is not Ellipsis else default

    def ranges(self):
        yield AnyRange()

    def bounds(self):
        return (None, None)

    def clear(self):
        return

    def __len__(self):
        raise TypeError("object of type 'Any' has no len()")

    @property
    def dimen(self):
        return None

    @property
    def domain(self):
        return Any

    def __str__(self):
        if self._name is not None:
            return self.name
        return type(self).__name__