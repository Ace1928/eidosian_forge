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
class ImpliedDoLoop(Token):
    """ Represents an implied do loop in Fortran.

    Examples
    ========

    >>> from sympy import Symbol, fcode
    >>> from sympy.codegen.fnodes import ImpliedDoLoop, ArrayConstructor
    >>> i = Symbol('i', integer=True)
    >>> idl = ImpliedDoLoop(i**3, i, -3, 3, 2)  # -27, -1, 1, 27
    >>> ac = ArrayConstructor([-28, idl, 28]) # -28, -27, -1, 1, 27, 28
    >>> fcode(ac, standard=2003, source_format='free')
    '[-28, (i**3, i = -3, 3, 2), 28]'

    """
    __slots__ = _fields = ('expr', 'counter', 'first', 'last', 'step')
    defaults = {'step': Integer(1)}
    _construct_expr = staticmethod(sympify)
    _construct_counter = staticmethod(sympify)
    _construct_first = staticmethod(sympify)
    _construct_last = staticmethod(sympify)
    _construct_step = staticmethod(sympify)