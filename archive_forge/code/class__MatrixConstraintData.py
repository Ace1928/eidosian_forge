from pyomo.common.dependencies import (
from pyomo.core.expr.numvalue import NumericValue, value
from pyomo.core.kernel.constraint import IConstraint, constraint_tuple
class _MatrixConstraintData(IConstraint):
    """
    A placeholder object for linear constraints in a
    matrix_constraint container. A user should not
    directly instantiate this class.
    """
    _ctype = IConstraint
    _linear_canonical_form = True
    __slots__ = ('_parent', '_storage_key', '_active', '__weakref__')

    def __init__(self, index):
        assert index >= 0
        self._parent = None
        self._storage_key = index
        self._active = True

    @property
    def index(self):
        """The row index of this constraint in the parent matrix"""
        return self._storage_key

    @property
    def terms(self):
        """An iterator over the terms in the body of this
        constraint as (variable, coefficient) tuples"""
        parent = self.parent
        x = parent.x
        if x is None:
            raise ValueError('No variable order has been assigned')
        A = parent._A
        if parent._sparse:
            for k in range(A.indptr[self._storage_key], A.indptr[self._storage_key + 1]):
                yield (x[A.indices[k]], A.data[k])
        else:
            for item in zip(x, A[self._storage_key, :].tolist()):
                yield item

    def __call__(self, exception=True):
        if self.parent.x is None:
            raise ValueError('No variable order has been assigned')
        try:
            return sum((c * v() for v, c in self.terms))
        except (ValueError, TypeError):
            if exception:
                raise
            return None

    @property
    def body(self):
        """The body of the constraint"""
        return sum((c * v for v, c in self.terms))

    @property
    def lower(self):
        """The expression for the lower bound of the constraint"""
        return self.parent.lb[self._storage_key]

    @lower.setter
    def lower(self, lb):
        if self.equality:
            raise ValueError('The lower property can not be set when the equality property is True.')
        if lb is None:
            lb = -numpy.inf
        elif isinstance(lb, NumericValue):
            raise ValueError('lb must be set to a simple numeric type or None')
        self.parent.lb[self._storage_key] = lb

    @property
    def upper(self):
        """The expression for the upper bound of the constraint"""
        return self.parent.ub[self._storage_key]

    @upper.setter
    def upper(self, ub):
        if self.equality:
            raise ValueError('The upper property can not be set when the equality property is True.')
        if ub is None:
            ub = numpy.inf
        elif isinstance(ub, NumericValue):
            raise ValueError('ub must be set to a simple numeric type or None')
        self.parent.ub[self._storage_key] = ub

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
        """The right-hand side of the constraint. This
        property can only be read when the equality property
        is :const:`True`. Assigning to this property
        implicitly sets the equality property to
        :const:`True`."""
        if not self.equality:
            raise ValueError('The rhs property can not be read when the equality property is False.')
        return self.parent.lb[self._storage_key]

    @rhs.setter
    def rhs(self, rhs):
        if rhs is None:
            raise ValueError('Constraint right-hand side can not be assigned a value of None.')
        elif isinstance(rhs, NumericValue):
            raise ValueError('rhs must be set to a simple numeric type or None')
        self.parent.lb[self._storage_key] = rhs
        self.parent.ub[self._storage_key] = rhs
        self.parent.equality[self._storage_key] = True

    @property
    def bounds(self):
        """The bounds of the constraint as a tuple (lb, ub)"""
        return (self.parent.lb[self._storage_key], self.parent.ub[self._storage_key])

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
        return self.parent.equality[self._storage_key]

    @equality.setter
    def equality(self, equality):
        if equality:
            raise ValueError('The constraint equality flag can only be set to True by assigning a value to the rhs property (e.g., con.rhs = con.lb).')
        assert not equality
        self.parent.equality[self._storage_key] = False

    def canonical_form(self, compute_values=True):
        """Build a canonical representation of the body of
        this constraints"""
        from pyomo.repn.standard_repn import StandardRepn
        variables = []
        coefficients = []
        constant = 0
        for v, c in self.terms:
            c = float(c)
            if not v.fixed:
                variables.append(v)
                coefficients.append(c)
            elif compute_values:
                constant += c * v()
            else:
                constant += c * v
        repn = StandardRepn()
        repn.linear_vars = tuple(variables)
        repn.linear_coefs = tuple(coefficients)
        repn.constant = constant
        return repn