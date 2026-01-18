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
class Extent(Basic):
    """ Represents a dimension extent.

    Examples
    ========

    >>> from sympy.codegen.fnodes import Extent
    >>> e = Extent(-3, 3)  # -3, -2, -1, 0, 1, 2, 3
    >>> from sympy import fcode
    >>> fcode(e, source_format='free')
    '-3:3'
    >>> from sympy.codegen.ast import Variable, real
    >>> from sympy.codegen.fnodes import dimension, intent_out
    >>> dim = dimension(e, e)
    >>> arr = Variable('x', real, attrs=[dim, intent_out])
    >>> fcode(arr.as_Declaration(), source_format='free', standard=2003)
    'real*8, dimension(-3:3, -3:3), intent(out) :: x'

    """

    def __new__(cls, *args):
        if len(args) == 2:
            low, high = args
            return Basic.__new__(cls, sympify(low), sympify(high))
        elif len(args) == 0 or (len(args) == 1 and args[0] in (':', None)):
            return Basic.__new__(cls)
        else:
            raise ValueError("Expected 0 or 2 args (or one argument == None or ':')")

    def _sympystr(self, printer):
        if len(self.args) == 0:
            return ':'
        return ':'.join((str(arg) for arg in self.args))