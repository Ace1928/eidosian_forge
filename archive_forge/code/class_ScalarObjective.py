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
class ScalarObjective(_GeneralObjectiveData, Objective):
    """
    ScalarObjective is the implementation representing a single,
    non-indexed objective.
    """

    def __init__(self, *args, **kwd):
        _GeneralObjectiveData.__init__(self, expr=None, component=self)
        Objective.__init__(self, *args, **kwd)
        self._index = UnindexedComponent_index

    def __call__(self, exception=True):
        if self._constructed:
            if len(self._data) == 0:
                raise ValueError("Evaluating the expression of ScalarObjective '%s' before the Objective has been assigned a sense or expression (there is currently no value to return)." % self.name)
            return super().__call__(exception)
        raise ValueError("Evaluating the expression of objective '%s' before the Objective has been constructed (there is currently no value to return)." % self.name)

    @property
    def expr(self):
        """Access the expression of this objective."""
        if self._constructed:
            if len(self._data) == 0:
                raise ValueError("Accessing the expression of ScalarObjective '%s' before the Objective has been assigned a sense or expression (there is currently no value to return)." % self.name)
            return _GeneralObjectiveData.expr.fget(self)
        raise ValueError("Accessing the expression of objective '%s' before the Objective has been constructed (there is currently no value to return)." % self.name)

    @expr.setter
    def expr(self, expr):
        """Set the expression of this objective."""
        self.set_value(expr)

    @property
    def sense(self):
        """Access sense (direction) of this objective."""
        if self._constructed:
            if len(self._data) == 0:
                raise ValueError("Accessing the sense of ScalarObjective '%s' before the Objective has been assigned a sense or expression (there is currently no value to return)." % self.name)
            return _GeneralObjectiveData.sense.fget(self)
        raise ValueError("Accessing the sense of objective '%s' before the Objective has been constructed (there is currently no value to return)." % self.name)

    @sense.setter
    def sense(self, sense):
        """Set the sense (direction) of this objective."""
        self.set_sense(sense)

    def clear(self):
        self._data = {}

    def set_value(self, expr):
        """Set the expression of this objective."""
        if not self._constructed:
            raise ValueError("Setting the value of objective '%s' before the Objective has been constructed (there is currently no object to set)." % self.name)
        if not self._data:
            self._data[None] = self
        return super().set_value(expr)

    def set_sense(self, sense):
        """Set the sense (direction) of this objective."""
        if self._constructed:
            if len(self._data) == 0:
                self._data[None] = self
            return _GeneralObjectiveData.set_sense(self, sense)
        raise ValueError("Setting the sense of objective '%s' before the Objective has been constructed (there is currently no object to set)." % self.name)

    def add(self, index, expr):
        """Add an expression with a given index."""
        if index is not None:
            raise ValueError("ScalarObjective object '%s' does not accept index values other than None. Invalid value: %s" % (self.name, index))
        self.set_value(expr)
        return self