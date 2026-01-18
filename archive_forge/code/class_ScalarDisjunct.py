import logging
import sys
import types
from math import fabs
from weakref import ref as weakref_ref
from pyomo.common.autoslots import AutoSlots
from pyomo.common.deprecation import deprecation_warning, RenamedClass
from pyomo.common.errors import PyomoException
from pyomo.common.log import is_debug_set
from pyomo.common.numeric_types import native_logical_types, native_types
from pyomo.common.modeling import unique_component_name, NOTSET
from pyomo.common.timing import ConstructionTimer
from pyomo.core import (
from pyomo.core.base.component import (
from pyomo.core.base.global_set import UnindexedComponent_index
from pyomo.core.base.block import _BlockData
from pyomo.core.base.misc import apply_indexed_rule
from pyomo.core.base.indexed_component import ActiveIndexedComponent
from pyomo.core.expr.expr_common import ExpressionType
class ScalarDisjunct(_DisjunctData, Disjunct):

    def __init__(self, *args, **kwds):
        self._defer_construction = True
        self._suppress_ctypes = set()
        _DisjunctData.__init__(self, self)
        Disjunct.__init__(self, *args, **kwds)
        self._data[None] = self
        self._index = UnindexedComponent_index