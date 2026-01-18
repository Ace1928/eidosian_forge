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
class GoTo(Token):
    """ Represents a goto statement in Fortran

    Examples
    ========

    >>> from sympy.codegen.fnodes import GoTo
    >>> go = GoTo([10, 20, 30], 'i')
    >>> from sympy import fcode
    >>> fcode(go, source_format='free')
    'go to (10, 20, 30), i'

    """
    __slots__ = _fields = ('labels', 'expr')
    defaults = {'expr': none}
    _construct_labels = staticmethod(_mk_Tuple)
    _construct_expr = staticmethod(sympify)