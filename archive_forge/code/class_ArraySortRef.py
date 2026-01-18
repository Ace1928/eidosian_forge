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
class ArraySortRef(SortRef):
    """Array sorts."""

    def domain(self):
        """Return the domain of the array sort `self`.

        >>> A = ArraySort(IntSort(), BoolSort())
        >>> A.domain()
        Int
        """
        return _to_sort_ref(Z3_get_array_sort_domain(self.ctx_ref(), self.ast), self.ctx)

    def domain_n(self, i):
        """Return the domain of the array sort `self`.
        """
        return _to_sort_ref(Z3_get_array_sort_domain_n(self.ctx_ref(), self.ast, i), self.ctx)

    def range(self):
        """Return the range of the array sort `self`.

        >>> A = ArraySort(IntSort(), BoolSort())
        >>> A.range()
        Bool
        """
        return _to_sort_ref(Z3_get_array_sort_range(self.ctx_ref(), self.ast), self.ctx)