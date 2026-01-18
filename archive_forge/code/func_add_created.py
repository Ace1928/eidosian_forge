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
def add_created(self, created):
    assert not self.created
    assert not self._ctx
    if self.solver:
        Z3_solver_propagate_created(self.ctx_ref(), self.solver.solver, _user_prop_created)
    self.created = created