import dis
from contextlib import contextmanager
import builtins
import operator
from typing import Iterator
from functools import reduce
from numba.core import (
from numba.core.utils import (
from .rvsdg.bc2rvsdg import (
from .rvsdg.regionpasses import RegionVisitor
def _test_none_and_jump(self, pred, bc: dis.Instruction, invert: bool):
    test = 'is not' if invert else 'is'
    op = BINOPS_TO_OPERATORS[test]
    none = self.store(value=ir.Const(None, loc=self.loc), name=f'$constNone{bc.offset}')
    isnone = ir.Expr.binop(op, lhs=self.vsmap[pred], rhs=none, loc=self.loc)
    self.branch_predicate = self.store(isnone, '$jump_if')