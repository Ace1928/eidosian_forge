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
class FPRMRef(ExprRef):
    """Floating-point rounding mode expressions"""

    def as_string(self):
        """Return a Z3 floating point expression as a Python string."""
        return Z3_ast_to_string(self.ctx_ref(), self.as_ast())