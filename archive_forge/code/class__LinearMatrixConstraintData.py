import time
import logging
import array
from weakref import ref as weakref_ref
from pyomo.common.log import is_debug_set
from pyomo.common.numeric_types import value
from pyomo.core.expr.numvalue import is_fixed, ZeroConstant
from pyomo.core.base.set_types import Any
from pyomo.core.base import SortComponents, Var
from pyomo.core.base.component import ModelComponentFactory
from pyomo.core.base.constraint import (
from pyomo.core.expr.numvalue import native_numeric_types
from pyomo.repn import generate_standard_repn
from collections.abc import Mapping
class _LinearMatrixConstraintData(_LinearConstraintData):
    """
    This class defines the data for a single linear constraint
        derived from a canonical form Ax=b constraint.

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
        variables       A tuple of variables comprising the body
                            of this constraint
        coefficients    A tuple of coefficients matching the order
                            of variables that comprise the body of
                            this constraint
        constant        A number representing the aggregate of any
                            constants found in the body of this
                            constraint

    Private class attributes:
        _component      The objective component.
        _active         A boolean that indicates whether this data is active
    """
    __slots__ = ()

    def __init__(self, index, component=None):
        self._component = weakref_ref(component) if component is not None else None
        self._active = True
        assert index >= 0
        self._index = index

    def __getstate__(self):
        """
        This method must be defined because this class uses slots.
        """
        result = super(_LinearMatrixConstraintData, self).__getstate__()
        return result

    def __call__(self, exception=True):
        """
        Compute the value of the body of this constraint.
        """
        comp = self.parent_component()
        index = self.index()
        prows = comp._prows
        jcols = comp._jcols
        varmap = comp._varmap
        vals = comp._vals
        try:
            return sum((varmap[jcols[p]]() * vals[p] for p in range(prows[index], prows[index + 1])))
        except (ValueError, TypeError):
            if exception:
                raise
            return None

    def has_lb(self):
        """Returns :const:`False` when the lower bound is
        :const:`None` or negative infinity"""
        lb = self.lower
        return lb is not None and lb != float('-inf')

    def has_ub(self):
        """Returns :const:`False` when the upper bound is
        :const:`None` or positive infinity"""
        ub = self.upper
        return ub is not None and ub != float('inf')

    def lslack(self):
        """
        Returns the value of L-f(x) for constraints of the form:
            L <= f(x) (<= U)
            (U >=) f(x) >= L
        """
        raise self.lower - self()

    def uslack(self):
        """
        Returns the value of U-f(x) for constraints of the form:
            (L <=) f(x) <= U
            U >= f(x) (>= L)
        """
        self.upper - self()

    def index(self):
        return self._index

    @property
    def variables(self):
        """A tuple of variables comprising the constraint body."""
        comp = self.parent_component()
        prows = comp._prows
        jcols = comp._jcols
        varmap = comp._varmap
        if prows[self._index] == prows[self._index + 1]:
            return ()
        variables = tuple((varmap[jcols[p]] for p in range(prows[self._index], prows[self._index + 1]) if not varmap[jcols[p]].fixed))
        return variables

    @property
    def coefficients(self):
        """A tuple of coefficients associated with the variables."""
        comp = self.parent_component()
        prows = comp._prows
        jcols = comp._jcols
        vals = comp._vals
        varmap = comp._varmap
        if prows[self._index] == prows[self._index + 1]:
            return ()
        coefs = tuple((vals[p] for p in range(prows[self._index], prows[self._index + 1]) if not varmap[jcols[p]].fixed))
        return coefs
    linear = coefficients

    @property
    def constant(self):
        """The constant value associated with the constraint body."""
        comp = self.parent_component()
        prows = comp._prows
        jcols = comp._jcols
        vals = comp._vals
        varmap = comp._varmap
        if prows[self._index] == prows[self._index + 1]:
            return 0
        terms = tuple((vals[p] * varmap[jcols[p]]() for p in range(prows[self._index], prows[self._index + 1]) if varmap[jcols[p]].fixed))
        return sum(terms)

    @property
    def body(self):
        """Access the body of a constraint expression."""
        comp = self.parent_component()
        index = self.index()
        prows = comp._prows
        jcols = comp._jcols
        varmap = comp._varmap
        vals = comp._vals
        if prows[self._index] == prows[self._index + 1]:
            return ZeroConstant
        return sum((varmap[jcols[p]] * vals[p] for p in range(prows[index], prows[index + 1])))

    @property
    def lower(self):
        """Access the lower bound of a constraint expression."""
        comp = self.parent_component()
        index = self.index()
        if comp._range_types[index] & MatrixConstraint.LowerBound:
            return comp._ranges[2 * index]
        return None

    @property
    def upper(self):
        """Access the upper bound of a constraint expression."""
        comp = self.parent_component()
        index = self.index()
        if comp._range_types[index] & MatrixConstraint.UpperBound:
            return comp._ranges[2 * index + 1]
        return None

    @property
    def lb(self):
        """Access the lower bound of a constraint expression."""
        return self.lower

    @property
    def ub(self):
        """Access the upper bound of a constraint expression."""
        return self.upper

    @property
    def equality(self):
        """A boolean indicating whether this is an equality constraint."""
        return self.parent_component()._range_types[self.index()] & MatrixConstraint.Equality == MatrixConstraint.Equality

    @property
    def strict_lower(self):
        """A boolean indicating whether this constraint has a strict lower bound."""
        return self.parent_component()._range_types[self.index()] & MatrixConstraint.StrictLowerBound == MatrixConstraint.StrictLowerBound

    @property
    def strict_upper(self):
        """A boolean indicating whether this constraint has a strict upper bound."""
        return self.parent_component()._range_types[self.index()] & MatrixConstraint.StrictUpperBound == MatrixConstraint.StrictUpperBound

    def set_value(self, expr):
        """Set the expression on this constraint."""
        raise NotImplementedError('MatrixConstraint row elements can not be updated')