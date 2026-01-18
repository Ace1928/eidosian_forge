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
class FiniteDomainRef(ExprRef):
    """Finite-domain expressions."""

    def sort(self):
        """Return the sort of the finite-domain expression `self`."""
        return FiniteDomainSortRef(Z3_get_sort(self.ctx_ref(), self.as_ast()), self.ctx)

    def as_string(self):
        """Return a Z3 floating point expression as a Python string."""
        return Z3_ast_to_string(self.ctx_ref(), self.as_ast())