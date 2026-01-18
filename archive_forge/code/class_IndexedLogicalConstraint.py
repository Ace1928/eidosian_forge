import inspect
import sys
import logging
from weakref import ref as weakref_ref
from pyomo.common.deprecation import RenamedClass
from pyomo.common.formatting import tabular_writer
from pyomo.common.log import is_debug_set
from pyomo.common.modeling import NOTSET
from pyomo.common.timing import ConstructionTimer
from pyomo.core.expr.boolean_value import as_boolean, BooleanConstant
from pyomo.core.expr.numvalue import native_types, native_logical_types
from pyomo.core.base.component import ActiveComponentData, ModelComponentFactory
from pyomo.core.base.global_set import UnindexedComponent_index
from pyomo.core.base.indexed_component import (
from pyomo.core.base.misc import apply_indexed_rule
from pyomo.core.base.set import Set
class IndexedLogicalConstraint(LogicalConstraint):

    def add(self, index, expr):
        """Add a logical constraint with a given index."""
        return self.__setitem__(index, expr)