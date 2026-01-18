from sympy.core.add import Add
from sympy.core.containers import Tuple
from sympy.core.expr import Expr
from sympy.core.function import AppliedUndef, UndefinedFunction
from sympy.core.mul import Mul
from sympy.core.relational import Equality, Relational
from sympy.core.singleton import S
from sympy.core.symbol import Symbol, Dummy
from sympy.core.sympify import sympify
from sympy.functions.elementary.piecewise import (piecewise_fold,
from sympy.logic.boolalg import BooleanFunction
from sympy.matrices.matrices import MatrixBase
from sympy.sets.sets import Interval, Set
from sympy.sets.fancysets import Range
from sympy.tensor.indexed import Idx
from sympy.utilities import flatten
from sympy.utilities.iterables import sift, is_sequence
from sympy.utilities.exceptions import sympy_deprecation_warning

        Returns True if the limits are known to be in reversed order, either
        by the explicit bounds, assumptions on the bounds, or assumptions on the
        variables.  False if known to be in normal order, based on the bounds.
        None if not enough information is available to determine.

        Examples
        ========

        >>> from sympy import Sum, Integral, Product, oo, Symbol
        >>> x = Symbol('x')
        >>> Sum(x, (x, 8, 1)).has_reversed_limits
        True

        >>> Sum(x, (x, 1, oo)).has_reversed_limits
        False

        >>> M = Symbol('M')
        >>> Integral(x, (x, 1, M)).has_reversed_limits

        >>> N = Symbol('N', integer=True, positive=True)
        >>> Sum(x, (x, 1, N)).has_reversed_limits
        False

        >>> Product(x, (x, 2, N)).has_reversed_limits

        >>> Product(x, (x, 2, N)).subs(N, N + 2).has_reversed_limits
        False

        See Also
        ========

        sympy.concrete.expr_with_intlimits.ExprWithIntLimits.has_empty_sequence

        