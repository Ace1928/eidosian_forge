import sys
import logging
from weakref import ref as weakref_ref
from pyomo.common.pyomo_typing import overload
from pyomo.common.deprecation import RenamedClass
from pyomo.common.errors import DeveloperError
from pyomo.common.formatting import tabular_writer
from pyomo.common.log import is_debug_set
from pyomo.common.modeling import NOTSET
from pyomo.common.timing import ConstructionTimer
from pyomo.core.expr.numvalue import (
from pyomo.core.expr import (
from pyomo.core.base.component import ActiveComponentData, ModelComponentFactory
from pyomo.core.base.global_set import UnindexedComponent_index
from pyomo.core.base.indexed_component import (
from pyomo.core.base.set import Set
from pyomo.core.base.disable_methods import disable_methods
from pyomo.core.base.initializer import (
class _ConstraintData(ActiveComponentData):
    """
    This class defines the data for a single constraint.

    Constructor arguments:
        component       The Constraint object that owns this data.

    Public class attributes:
        active          A boolean that is true if this constraint is
                            active in the model.
        body            The Pyomo expression for this constraint
        lower           The Pyomo expression for the lower bound
        upper           The Pyomo expression for the upper bound
        equality        A boolean that indicates whether this is an
                            equality constraint
        strict_lower    A boolean that indicates whether this
                            constraint uses a strict lower bound
        strict_upper    A boolean that indicates whether this
                            constraint uses a strict upper bound

    Private class attributes:
        _component      The objective component.
        _active         A boolean that indicates whether this data is active
    """
    __slots__ = ()
    _linear_canonical_form = False

    def __init__(self, component=None):
        self._component = weakref_ref(component) if component is not None else None
        self._index = NOTSET
        self._active = True

    def __call__(self, exception=True):
        """Compute the value of the body of this constraint."""
        return value(self.body, exception=exception)

    def has_lb(self):
        """Returns :const:`False` when the lower bound is
        :const:`None` or negative infinity"""
        return self.lb is not None

    def has_ub(self):
        """Returns :const:`False` when the upper bound is
        :const:`None` or positive infinity"""
        return self.ub is not None

    def lslack(self):
        """
        Returns the value of f(x)-L for constraints of the form:
            L <= f(x) (<= U)
            (U >=) f(x) >= L
        """
        lb = self.lb
        if lb is None:
            return _inf
        else:
            return value(self.body) - lb

    def uslack(self):
        """
        Returns the value of U-f(x) for constraints of the form:
            (L <=) f(x) <= U
            U >= f(x) (>= L)
        """
        ub = self.ub
        if ub is None:
            return _inf
        else:
            return ub - value(self.body)

    def slack(self):
        """
        Returns the smaller of lslack and uslack values
        """
        lb = self.lb
        ub = self.ub
        body = value(self.body)
        if lb is None:
            return ub - body
        elif ub is None:
            return body - lb
        return min(ub - body, body - lb)

    @property
    def body(self):
        """Access the body of a constraint expression."""
        raise NotImplementedError

    @property
    def lower(self):
        """Access the lower bound of a constraint expression."""
        raise NotImplementedError

    @property
    def upper(self):
        """Access the upper bound of a constraint expression."""
        raise NotImplementedError

    @property
    def lb(self):
        """Access the value of the lower bound of a constraint expression."""
        raise NotImplementedError

    @property
    def ub(self):
        """Access the value of the upper bound of a constraint expression."""
        raise NotImplementedError

    @property
    def equality(self):
        """A boolean indicating whether this is an equality constraint."""
        raise NotImplementedError

    @property
    def strict_lower(self):
        """True if this constraint has a strict lower bound."""
        raise NotImplementedError

    @property
    def strict_upper(self):
        """True if this constraint has a strict upper bound."""
        raise NotImplementedError

    def set_value(self, expr):
        """Set the expression on this constraint."""
        raise NotImplementedError

    def get_value(self):
        """Get the expression on this constraint."""
        raise NotImplementedError