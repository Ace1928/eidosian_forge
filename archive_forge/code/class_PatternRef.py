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
class PatternRef(ExprRef):
    """Patterns are hints for quantifier instantiation.

    """

    def as_ast(self):
        return Z3_pattern_to_ast(self.ctx_ref(), self.ast)

    def get_id(self):
        return Z3_get_ast_id(self.ctx_ref(), self.as_ast())