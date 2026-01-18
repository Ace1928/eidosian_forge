from sympy.codegen.ast import (
from sympy.core.basic import Basic
from sympy.core.containers import Tuple
from sympy.core.sympify import sympify
class PreDecrement(Basic):
    """ Represents the pre-decrement operator

    Examples
    ========

    >>> from sympy.abc import x
    >>> from sympy.codegen.cnodes import PreDecrement
    >>> from sympy import ccode
    >>> ccode(PreDecrement(x))
    '--(x)'

    """
    nargs = 1