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
class ScalarDisjunction(_DisjunctionData, Disjunction):

    def __init__(self, *args, **kwds):
        _DisjunctionData.__init__(self, component=self)
        Disjunction.__init__(self, *args, **kwds)
        self._index = UnindexedComponent_index

    def set_value(self, expr):
        """Set the expression on this disjunction."""
        if not self._constructed:
            raise ValueError("Setting the value of disjunction '%s' before the Disjunction has been constructed (there is currently no object to set)." % self.name)
        if len(self._data) == 0:
            self._data[None] = self
        if expr is Disjunction.Skip:
            del self[None]
            return None
        return super(ScalarDisjunction, self).set_value(expr)