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
class FiniteDomainSortRef(SortRef):
    """Finite domain sort."""

    def size(self):
        """Return the size of the finite domain sort"""
        r = (ctypes.c_ulonglong * 1)()
        if Z3_get_finite_domain_sort_size(self.ctx_ref(), self.ast, r):
            return r[0]
        else:
            raise Z3Exception('Failed to retrieve finite domain sort size')