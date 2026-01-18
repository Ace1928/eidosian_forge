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
class ScalarLogicalConstraint(_GeneralLogicalConstraintData, LogicalConstraint):
    """
    ScalarLogicalConstraint is the implementation representing a single,
    non-indexed logical constraint.
    """

    def __init__(self, *args, **kwds):
        _GeneralLogicalConstraintData.__init__(self, component=self, expr=None)
        LogicalConstraint.__init__(self, *args, **kwds)
        self._index = UnindexedComponent_index

    @property
    def body(self):
        """Access the body of a logical constraint."""
        if self._constructed:
            if len(self._data) == 0:
                raise ValueError("Accessing the body of ScalarLogicalConstraint '%s' before the LogicalConstraint has been assigned an expression. There is currently nothing to access." % self.name)
            return _GeneralLogicalConstraintData.body.fget(self)
        raise ValueError("Accessing the body of logical constraint '%s' before the LogicalConstraint has been constructed (there is currently no value to return)." % self.name)

    def set_value(self, expr):
        """Set the expression on this logical constraint."""
        if not self._constructed:
            raise ValueError("Setting the value of logical constraint '%s' before the LogicalConstraint has been constructed (there is currently no object to set)." % self.name)
        if len(self._data) == 0:
            self._data[None] = self
        if self._check_skip_add(None, expr) is None:
            del self[None]
            return None
        return super(ScalarLogicalConstraint, self).set_value(expr)

    def add(self, index, expr):
        """Add a logical constraint with a given index."""
        if index is not None:
            raise ValueError("ScalarLogicalConstraint object '%s' does not accept index values other than None. Invalid value: %s" % (self.name, index))
        self.set_value(expr)
        return self