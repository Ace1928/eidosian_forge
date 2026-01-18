from pyomo.core.expr.numvalue import (
from pyomo.core.expr.expr_common import ExpressionType
from pyomo.core.expr.relational_expr import (
from pyomo.core.kernel.base import ICategorizedObject, _abstract_readonly_property
from pyomo.core.kernel.container_utils import define_simple_containers
class _MutableBoundsConstraintMixin(object):
    """
    Use as a base class for IConstraint implementations
    that allow adjusting the lb, ub, rhs, and equality
    properties.

    Assumes the derived class has _lb, _ub, and _equality
    attributes that can be modified.
    """
    __slots__ = ()

    @property
    def lower(self):
        """The expression for the lower bound of the constraint"""
        return self._lb

    @lower.setter
    def lower(self, lb):
        if self.equality:
            raise ValueError('The lower property can not be set when the equality property is True.')
        if lb is not None and (not is_numeric_data(lb)):
            raise TypeError('Constraint lower bounds must be expressions restricted to numeric data.')
        self._lb = lb

    @property
    def upper(self):
        """The expression for the upper bound of the constraint"""
        return self._ub

    @upper.setter
    def upper(self, ub):
        if self.equality:
            raise ValueError('The upper property can not be set when the equality property is True.')
        if ub is not None and (not is_numeric_data(ub)):
            raise TypeError('Constraint upper bounds must be expressions restricted to numeric data.')
        self._ub = ub

    @property
    def lb(self):
        """The value of the lower bound of the constraint"""
        lb = value(self.lower)
        if lb == _neg_inf:
            return None
        return lb

    @lb.setter
    def lb(self, lb):
        self.lower = lb

    @property
    def ub(self):
        """The value of the upper bound of the constraint"""
        ub = value(self.upper)
        if ub == _pos_inf:
            return None
        return ub

    @ub.setter
    def ub(self, ub):
        self.upper = ub

    @property
    def rhs(self):
        """The right-hand side of the constraint"""
        if not self.equality:
            raise ValueError('The rhs property can not be read when the equality property is False.')
        return self._lb

    @rhs.setter
    def rhs(self, rhs):
        if rhs is None:
            raise ValueError('Constraint right-hand side can not be assigned a value of None.')
        elif not is_numeric_data(rhs):
            raise TypeError('Constraint right-hand side must be numbers or expressions restricted to data.')
        self._lb = rhs
        self._ub = rhs
        self._equality = True

    @property
    def bounds(self):
        """The bounds of the constraint as a tuple (lb, ub)"""
        return super(_MutableBoundsConstraintMixin, self).bounds

    @bounds.setter
    def bounds(self, bounds_tuple):
        self.lb, self.ub = bounds_tuple

    @property
    def equality(self):
        """Returns :const:`True` when this is an equality
        constraint.

        Disable equality by assigning
        :const:`False`. Equality can only be activated by
        assigning a value to the .rhs property."""
        return self._equality

    @equality.setter
    def equality(self, equality):
        if equality:
            raise ValueError('The constraint equality flag can only be set to True by assigning a value to the rhs property (e.g., con.rhs = con.lb).')
        assert not equality
        self._equality = False