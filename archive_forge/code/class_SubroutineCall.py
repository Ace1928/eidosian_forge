from sympy.codegen.ast import (
from sympy.core.basic import Basic
from sympy.core.containers import Tuple
from sympy.core.expr import Expr
from sympy.core.function import Function
from sympy.core.numbers import Float, Integer
from sympy.core.symbol import Str
from sympy.core.sympify import sympify
from sympy.logic import true, false
from sympy.utilities.iterables import iterable
class SubroutineCall(Token):
    """ Represents a call to a subroutine in Fortran.

    Examples
    ========

    >>> from sympy.codegen.fnodes import SubroutineCall
    >>> from sympy import fcode
    >>> fcode(SubroutineCall('mysub', 'x y'.split()))
    '       call mysub(x, y)'

    """
    __slots__ = _fields = ('name', 'subroutine_args')
    _construct_name = staticmethod(_name)
    _construct_subroutine_args = staticmethod(_mk_Tuple)