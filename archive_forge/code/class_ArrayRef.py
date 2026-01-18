from . import z3core
from .z3core import *
from .z3types import *
from .z3consts import *
from .z3printer import *
from fractions import Fraction
import sys
import io
import math
import copy
class ArrayRef(ExprRef):
    """Array expressions. """

    def sort(self):
        """Return the array sort of the array expression `self`.

        >>> a = Array('a', IntSort(), BoolSort())
        >>> a.sort()
        Array(Int, Bool)
        """
        return ArraySortRef(Z3_get_sort(self.ctx_ref(), self.as_ast()), self.ctx)

    def domain(self):
        """Shorthand for `self.sort().domain()`.

        >>> a = Array('a', IntSort(), BoolSort())
        >>> a.domain()
        Int
        """
        return self.sort().domain()

    def domain_n(self, i):
        """Shorthand for self.sort().domain_n(i)`."""
        return self.sort().domain_n(i)

    def range(self):
        """Shorthand for `self.sort().range()`.

        >>> a = Array('a', IntSort(), BoolSort())
        >>> a.range()
        Bool
        """
        return self.sort().range()

    def __getitem__(self, arg):
        """Return the Z3 expression `self[arg]`.

        >>> a = Array('a', IntSort(), BoolSort())
        >>> i = Int('i')
        >>> a[i]
        a[i]
        >>> a[i].sexpr()
        '(select a i)'
        """
        return _array_select(self, arg)

    def default(self):
        return _to_expr_ref(Z3_mk_array_default(self.ctx_ref(), self.as_ast()), self.ctx)