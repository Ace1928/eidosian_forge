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
class _LogicalConstraintData(ActiveComponentData):
    """
    This class defines the data for a single logical constraint.

    It functions as a pure interface.

    Constructor arguments:
        component       The LogicalConstraint object that owns this data.

    Public class attributes:
        active          A boolean that is true if this statement is
                            active in the model.
        body            The Pyomo logical expression for this statement

    Private class attributes:
        _component      The statement component.
        _active         A boolean that indicates whether this data is active
    """
    __slots__ = ()

    def __init__(self, component=None):
        self._component = weakref_ref(component) if component is not None else None
        self._index = NOTSET
        self._active = True

    def __call__(self, exception=True):
        """Compute the value of the body of this logical constraint."""
        if self.body is None:
            return None
        return self.body(exception=exception)

    @property
    def expr(self):
        """Get the expression on this logical constraint."""
        raise NotImplementedError

    def set_value(self, expr):
        """Set the expression on this logical constraint."""
        raise NotImplementedError

    def get_value(self):
        """Get the expression on this logical constraint."""
        raise NotImplementedError