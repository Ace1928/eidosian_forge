from __future__ import annotations
from sympy.core import S
from sympy.core.expr import Expr
from sympy.core.symbol import Symbol, symbols as _symbols
from sympy.core.sympify import CantSympify
from sympy.printing.defaults import DefaultPrinting
from sympy.utilities import public
from sympy.utilities.iterables import flatten, is_sequence
from sympy.utilities.magic import pollute
from sympy.utilities.misc import as_int

        Check if `self == other**n` for some integer n.

        Examples
        ========

        >>> from sympy.combinatorics import free_group
        >>> F, x, y = free_group("x, y")
        >>> ((x*y)**2).power_of(x*y)
        True
        >>> (x**-3*y**-2*x**3).power_of(x**-3*y*x**3)
        True

        