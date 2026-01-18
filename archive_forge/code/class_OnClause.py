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
class OnClause:

    def __init__(self, s, on_clause):
        self.s = s
        self.ctx = s.ctx
        self.on_clause = on_clause
        self.idx = 22
        global _my_hacky_class
        _my_hacky_class = self
        Z3_solver_register_on_clause(self.ctx.ref(), self.s.solver, self.idx, _on_clause_eh)