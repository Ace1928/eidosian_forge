from pyomo.core.expr.numvalue import is_numeric_data
from pyomo.core.expr import value, exp
from pyomo.core.kernel.block import block
from pyomo.core.kernel.variable import IVariable, variable, variable_tuple
from pyomo.core.kernel.constraint import (
class quadratic(_ConicBase):
    """A quadratic conic constraint of the form:

        x[0]^2 + ... + x[n-1]^2 <= r^2,

    which is recognized as convex for r >= 0.

    Parameters
    ----------
    r : :class:`variable`
        A variable.
    x : list[:class:`variable`]
        An iterable of variables.
    """
    __slots__ = ('_parent', '_storage_key', '_active', '_body', '_r', '_x', '__weakref__')

    def __init__(self, r, x):
        super(quadratic, self).__init__()
        self._r = r
        self._x = tuple(x)
        assert isinstance(self._r, IVariable)
        assert all((isinstance(xi, IVariable) for xi in self._x))

    @classmethod
    def as_domain(cls, r, x):
        """Builds a conic domain. Input arguments take the
        same form as those of the conic constraint, but in
        place of each variable, one can optionally supply a
        constant, linear expression, or None.

        Returns
        -------
        block
            A block object with the core conic constraint
            (block.q) expressed using auxiliary variables
            (block.r, block.x) linked to the input arguments
            through auxiliary constraints (block.c).
        """
        b = block()
        b.r = variable(lb=0)
        b.x = variable_tuple([variable() for i in range(len(x))])
        b.c = _build_linking_constraints([r] + list(x), [b.r] + list(b.x))
        b.q = cls(r=b.r, x=b.x)
        return b

    @property
    def r(self):
        return self._r

    @property
    def x(self):
        return self._x

    def _body_function(self, r, x):
        """A function that defines the body expression"""
        return sum((xi ** 2 for xi in x)) - r ** 2

    def _body_function_variables(self, values=False):
        """Returns variables in the order they should be
        passed to the body function. If values is True, then
        return the current value of each variable in place
        of the variables themselves."""
        if not values:
            return (self.r, self.x)
        else:
            return (self.r.value, tuple((xi.value for xi in self.x)))

    def check_convexity_conditions(self, relax=False):
        """Returns True if all convexity conditions for the
        conic constraint are satisfied. If relax is True,
        then variable domains are ignored and it is assumed
        that all variables are continuous."""
        return (relax or (self.r.is_continuous() and all((xi.is_continuous() for xi in self.x)))) and (self.r.has_lb() and value(self.r.lb) >= 0)