import builtins
import collections
import dis
import operator
import logging
import textwrap
from numba.core import errors, ir, config
from numba.core.errors import NotDefinedError, UnsupportedError, error_extras
from numba.core.ir_utils import get_definition, guard
from numba.core.utils import (PYVERSION, BINOPS_TO_OPERATORS,
from numba.core.byteflow import Flow, AdaptDFA, AdaptCFA, BlockKind
from numba.core.unsafe import eh
from numba.cpython.unsafe.tuple import unpack_single_tuple
def _jump_if_none(self, inst, pred, iftrue):
    truebr = inst.next
    falsebr = inst.get_jump_target()
    if not iftrue:
        op = BINOPS_TO_OPERATORS['is']
    else:
        op = BINOPS_TO_OPERATORS['is not']
    rhs = self.store(value=ir.Const(None, loc=self.loc), name=f'$constNone{inst.offset}')
    lhs = self.get(pred)
    isnone = ir.Expr.binop(op, lhs=lhs, rhs=rhs, loc=self.loc)
    maybeNone = f'$maybeNone{inst.offset}'
    self.store(value=isnone, name=maybeNone)
    name = f'$bool{inst.offset}'
    gv_fn = ir.Global('bool', bool, loc=self.loc)
    self.store(value=gv_fn, name=name)
    callres = ir.Expr.call(self.get(name), (self.get(maybeNone),), (), loc=self.loc)
    pname = f'$pred{inst.offset}'
    predicate = self.store(value=callres, name=pname)
    branch = ir.Branch(cond=predicate, truebr=truebr, falsebr=falsebr, loc=self.loc)
    self.current_block.append(branch)