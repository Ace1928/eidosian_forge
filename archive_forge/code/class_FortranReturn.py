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
class FortranReturn(Token):
    """ AST node explicitly mapped to a fortran "return".

    Explanation
    ===========

    Because a return statement in fortran is different from C, and
    in order to aid reuse of our codegen ASTs the ordinary
    ``.codegen.ast.Return`` is interpreted as assignment to
    the result variable of the function. If one for some reason needs
    to generate a fortran RETURN statement, this node should be used.

    Examples
    ========

    >>> from sympy.codegen.fnodes import FortranReturn
    >>> from sympy import fcode
    >>> fcode(FortranReturn('x'))
    '       return x'

    """
    __slots__ = _fields = ('return_value',)
    defaults = {'return_value': none}
    _construct_return_value = staticmethod(sympify)