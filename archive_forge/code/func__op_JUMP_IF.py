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
def _op_JUMP_IF(self, inst, pred, iftrue):
    brs = {True: inst.get_jump_target(), False: inst.next}
    truebr = brs[iftrue]
    falsebr = brs[not iftrue]
    name = 'bool%s' % inst.offset
    gv_fn = ir.Global('bool', bool, loc=self.loc)
    self.store(value=gv_fn, name=name)
    callres = ir.Expr.call(self.get(name), (self.get(pred),), (), loc=self.loc)
    pname = '$%spred' % inst.offset
    predicate = self.store(value=callres, name=pname)
    bra = ir.Branch(cond=predicate, truebr=truebr, falsebr=falsebr, loc=self.loc)
    self.current_block.append(bra)