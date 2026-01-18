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
class BoundsInitializer(InitializerBase):
    """An Initializer wrapper that converts bounds information to a RangeSet

    The BoundsInitializer wraps another initializer that is expected to
    return valid arguments to the RangeSet constructor.  Nominally, this
    would be bounds information in the form of (lower bound, upper
    bound), but could also be a single scalar or a 3-tuple.  Calling
    this initializer will return a RangeSet object.

    BoundsInitializer objects can be intersected with other
    SetInitializer objects using the SetInitializer.intersect() method.

    """
    __slots__ = ('_init', 'default_step')

    def __init__(self, init, default_step=0):
        self._init = Initializer(init, treat_sequences_as_mappings=False)
        self.default_step = default_step

    def __call__(self, parent, idx):
        val = self._init(parent, idx)
        if not isinstance(val, Sequence):
            val = (1, val, self.default_step)
        else:
            val = tuple(val)
            if len(val) == 2:
                val += (self.default_step,)
            elif len(val) == 1:
                val = (1, val[0], self.default_step)
            elif len(val) == 0:
                val = (None, None, self.default_step)
        ans = RangeSet(*val)
        return ans

    def constant(self):
        return self._init.constant()

    def setdefault(self, val):
        pass