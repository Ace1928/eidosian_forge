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
class ArrayConstructor(Token):
    """ Represents an array constructor.

    Examples
    ========

    >>> from sympy import fcode
    >>> from sympy.codegen.fnodes import ArrayConstructor
    >>> ac = ArrayConstructor([1, 2, 3])
    >>> fcode(ac, standard=95, source_format='free')
    '(/1, 2, 3/)'
    >>> fcode(ac, standard=2003, source_format='free')
    '[1, 2, 3]'

    """
    __slots__ = _fields = ('elements',)
    _construct_elements = staticmethod(_mk_Tuple)