from sympy.core.function import Derivative
from sympy.core.singleton import S
from sympy.core.function import Subs
from sympy.core.traversal import preorder_traversal
from sympy.utilities.exceptions import sympy_deprecation_warning
from sympy.utilities.iterables import iterable
 Differentiate expr and replace Derivatives with finite differences.

    Parameters
    ==========

    expr : expression
    \*symbols : differentiate with respect to symbols
    points: sequence, coefficient or undefined function, optional
        see ``Derivative.as_finite_difference``
    x0: number or Symbol, optional
        see ``Derivative.as_finite_difference``
    wrt: Symbol, optional
        see ``Derivative.as_finite_difference``

    Examples
    ========

    >>> from sympy import sin, Function, differentiate_finite
    >>> from sympy.abc import x, y, h
    >>> f, g = Function('f'), Function('g')
    >>> differentiate_finite(f(x)*g(x), x, points=[x-h, x+h])
    -f(-h + x)*g(-h + x)/(2*h) + f(h + x)*g(h + x)/(2*h)

    ``differentiate_finite`` works on any expression, including the expressions
    with embedded derivatives:

    >>> differentiate_finite(f(x) + sin(x), x, 2)
    -2*f(x) + f(x - 1) + f(x + 1) - 2*sin(x) + sin(x - 1) + sin(x + 1)
    >>> differentiate_finite(f(x, y), x, y)
    f(x - 1/2, y - 1/2) - f(x - 1/2, y + 1/2) - f(x + 1/2, y - 1/2) + f(x + 1/2, y + 1/2)
    >>> differentiate_finite(f(x)*g(x).diff(x), x)
    (-g(x) + g(x + 1))*f(x + 1/2) - (g(x) - g(x - 1))*f(x - 1/2)

    To make finite difference with non-constant discretization step use
    undefined functions:

    >>> dx = Function('dx')
    >>> differentiate_finite(f(x)*g(x).diff(x), points=dx(x))
    -(-g(x - dx(x)/2 - dx(x - dx(x)/2)/2)/dx(x - dx(x)/2) +
    g(x - dx(x)/2 + dx(x - dx(x)/2)/2)/dx(x - dx(x)/2))*f(x - dx(x)/2)/dx(x) +
    (-g(x + dx(x)/2 - dx(x + dx(x)/2)/2)/dx(x + dx(x)/2) +
    g(x + dx(x)/2 + dx(x + dx(x)/2)/2)/dx(x + dx(x)/2))*f(x + dx(x)/2)/dx(x)

    