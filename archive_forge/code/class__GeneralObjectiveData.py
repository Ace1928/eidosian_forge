import sys
import logging
from weakref import ref as weakref_ref
from pyomo.common.pyomo_typing import overload
from pyomo.common.deprecation import RenamedClass
from pyomo.common.log import is_debug_set
from pyomo.common.modeling import NOTSET
from pyomo.common.formatting import tabular_writer
from pyomo.common.timing import ConstructionTimer
from pyomo.core.expr.numvalue import value
from pyomo.core.base.component import ActiveComponentData, ModelComponentFactory
from pyomo.core.base.global_set import UnindexedComponent_index
from pyomo.core.base.indexed_component import (
from pyomo.core.base.expression import _ExpressionData, _GeneralExpressionDataImpl
from pyomo.core.base.set import Set
from pyomo.core.base.initializer import (
from pyomo.core.base import minimize, maximize
class _GeneralObjectiveData(_GeneralExpressionDataImpl, _ObjectiveData, ActiveComponentData):
    """
    This class defines the data for a single objective.

    Note that this is a subclass of NumericValue to allow
    objectives to be used as part of expressions.

    Constructor arguments:
        expr            The Pyomo expression stored in this objective.
        sense           The direction for this objective.
        component       The Objective object that owns this data.

    Public class attributes:
        expr            The Pyomo expression for this objective
        active          A boolean that is true if this objective is active
                            in the model.
        sense           The direction for this objective.

    Private class attributes:
        _component      The objective component.
        _active         A boolean that indicates whether this data is active
    """
    __slots__ = ('_sense', '_args_')

    def __init__(self, expr=None, sense=minimize, component=None):
        _GeneralExpressionDataImpl.__init__(self, expr)
        self._component = weakref_ref(component) if component is not None else None
        self._index = NOTSET
        self._active = True
        self._sense = sense
        if self._sense != minimize and self._sense != maximize:
            raise ValueError("Objective sense must be set to one of 'minimize' (%s) or 'maximize' (%s). Invalid value: %s'" % (minimize, maximize, sense))

    def set_value(self, expr):
        if expr is None:
            raise ValueError(_rule_returned_none_error % (self.name,))
        return super().set_value(expr)

    @property
    def sense(self):
        """Access sense (direction) of this objective."""
        return self._sense

    @sense.setter
    def sense(self, sense):
        """Set the sense (direction) of this objective."""
        self.set_sense(sense)

    def set_sense(self, sense):
        """Set the sense (direction) of this objective."""
        if sense in {minimize, maximize}:
            self._sense = sense
        else:
            raise ValueError("Objective sense must be set to one of 'minimize' (%s) or 'maximize' (%s). Invalid value: %s'" % (minimize, maximize, sense))