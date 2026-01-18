from pyomo.core.expr.numvalue import is_numeric_data
from pyomo.core.expr import value, exp
from pyomo.core.kernel.block import block
from pyomo.core.kernel.variable import IVariable, variable, variable_tuple
from pyomo.core.kernel.constraint import (
class primal_power(_ConicBase):
    """A primal power conic constraint of the form:
       sqrt(x[0]^2 + ... + x[n-1]^2) <= (r1^alpha)*(r2^(1-alpha))

    which is recognized as convex for r1,r2 >= 0
    and 0 < alpha < 1.

    Parameters
    ----------
    r1 : :class:`variable`
        A variable.
    r2 : :class:`variable`
        A variable.
    x : list[:class:`variable`]
        An iterable of variables.
    alpha : float, :class:`parameter`, etc.
        A constant term.
    """
    __slots__ = ('_parent', '_storage_key', '_active', '_body', '_r1', '_r2', '_x', '_alpha', '__weakref__')

    def __init__(self, r1, r2, x, alpha):
        super(primal_power, self).__init__()
        self._r1 = r1
        self._r2 = r2
        self._x = tuple(x)
        self._alpha = alpha
        assert isinstance(self._r1, IVariable)
        assert isinstance(self._r2, IVariable)
        assert all((isinstance(xi, IVariable) for xi in self._x))
        if not is_numeric_data(self._alpha):
            raise TypeError('The type of the alpha parameter of a conic constraint is restricted numeric data or objects that store numeric data.')

    @classmethod
    def as_domain(cls, r1, r2, x, alpha):
        """Builds a conic domain. Input arguments take the
        same form as those of the conic constraint, but in
        place of each variable, one can optionally supply a
        constant, linear expression, or None.

        Returns
        -------
        block
            A block object with the core conic constraint
            (block.q) expressed using auxiliary variables
            (block.r1, block.r2, block.x) linked to the
            input arguments through auxiliary constraints
            (block.c).
        """
        b = block()
        b.r1 = variable(lb=0)
        b.r2 = variable(lb=0)
        b.x = variable_tuple([variable() for i in range(len(x))])
        b.c = _build_linking_constraints([r1, r2] + list(x), [b.r1, b.r2] + list(b.x))
        b.q = cls(r1=b.r1, r2=b.r2, x=b.x, alpha=alpha)
        return b

    @property
    def r1(self):
        return self._r1

    @property
    def r2(self):
        return self._r2

    @property
    def x(self):
        return self._x

    @property
    def alpha(self):
        return self._alpha

    def _body_function(self, r1, r2, x):
        """A function that defines the body expression"""
        alpha = self.alpha
        return sum((xi ** 2 for xi in x)) ** 0.5 - r1 ** alpha * r2 ** (1 - alpha)

    def _body_function_variables(self, values=False):
        """Returns variables in the order they should be
        passed to the body function. If values is True, then
        return the current value of each variable in place
        of the variables themselves."""
        if not values:
            return (self.r1, self.r2, self.x)
        else:
            return (self.r1.value, self.r2.value, tuple((xi.value for xi in self.x)))

    def check_convexity_conditions(self, relax=False):
        """Returns True if all convexity conditions for the
        conic constraint are satisfied. If relax is True,
        then variable domains are ignored and it is assumed
        that all variables are continuous."""
        alpha = value(self.alpha, exception=False)
        return (relax or (self.r1.is_continuous() and self.r2.is_continuous() and all((xi.is_continuous() for xi in self.x)))) and (self.r1.has_lb() and value(self.r1.lb) >= 0) and (self.r2.has_lb() and value(self.r2.lb) >= 0) and (alpha is not None and 0 < alpha < 1)