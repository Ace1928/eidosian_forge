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
class DatatypeRef(ExprRef):
    """Datatype expressions."""

    def sort(self):
        """Return the datatype sort of the datatype expression `self`."""
        return DatatypeSortRef(Z3_get_sort(self.ctx_ref(), self.as_ast()), self.ctx)