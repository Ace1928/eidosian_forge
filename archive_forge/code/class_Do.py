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
class Do(Token):
    """ Represents a Do loop in in Fortran.

    Examples
    ========

    >>> from sympy import fcode, symbols
    >>> from sympy.codegen.ast import aug_assign, Print
    >>> from sympy.codegen.fnodes import Do
    >>> i, n = symbols('i n', integer=True)
    >>> r = symbols('r', real=True)
    >>> body = [aug_assign(r, '+', 1/i), Print([i, r])]
    >>> do1 = Do(body, i, 1, n)
    >>> print(fcode(do1, source_format='free'))
    do i = 1, n
        r = r + 1d0/i
        print *, i, r
    end do
    >>> do2 = Do(body, i, 1, n, 2)
    >>> print(fcode(do2, source_format='free'))
    do i = 1, n, 2
        r = r + 1d0/i
        print *, i, r
    end do

    """
    __slots__ = _fields = ('body', 'counter', 'first', 'last', 'step', 'concurrent')
    defaults = {'step': Integer(1), 'concurrent': false}
    _construct_body = staticmethod(lambda body: CodeBlock(*body))
    _construct_counter = staticmethod(sympify)
    _construct_first = staticmethod(sympify)
    _construct_last = staticmethod(sympify)
    _construct_step = staticmethod(sympify)
    _construct_concurrent = staticmethod(lambda arg: true if arg else false)