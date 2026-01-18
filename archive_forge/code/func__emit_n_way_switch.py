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
def _emit_n_way_switch(self, block):
    """
        This handles emitting a switch block with N cases. It does the
        following:

        - The branch value table in the block provides information about the
          case index and the case label in this switch.
        - Emit a block that unconditionally jumps to the first case block.
        - In each case block:
            - Compare the control variable to the expected case index for
              that case.
            - Branch to the target label on true, or the next case on false.
        - There is no default case. The control variable must match a case index

        ┌───────────────────────┐
        │  current block        │
        └───────────┬───────────┘
                    │
                    └─────────┐
                              ▼
                    ┌───────────────────┐
                    │   case 0          │
                    └─────────┬─────────┘
                              │
                              │
                              ▼
                    ┌───────────────────┐
                    │  case 1           │
                    └─────────┬─────────┘
                              │
                              │
                              ▼
                    ┌───────────────────┐
                    │  case N-1         │
                    └─────────┬─────────┘
                              │
                    ┌─────────┘
                    │
                    ▼
        ┌───────────────────────┐
        │ subsequent blocks     │
        └───────────────────────┘

        """
    bvt = block.branch_value_table
    cp = block.variable
    cpvar = self.local_scope.get_exact(self._get_cp_name(cp))
    labels = [(k, self._get_label(v)) for k, v in bvt.items()]
    blocks = []
    for _ in range(len(labels) - 1):
        blocks.append((self._get_temp_label(), ir.Block(scope=self.local_scope, loc=self.loc)))
    with self.set_block(self._get_label(block.name), ir.Block(scope=self.local_scope, loc=self.loc)):
        self.current_block.append(ir.Jump(blocks[-1][0], loc=self.loc))
    while blocks:
        cp_expect, cp_label = labels.pop()
        cur_label, cur_block = blocks.pop()
        with self.set_block(cur_label, cur_block):
            const = self.store(ir.Const(cp_expect, loc=self.loc), '$.const')
            cmp = ir.Expr.binop(operator.eq, const, cpvar, loc=self.loc)
            pred = self.store(cmp, '$.cmp')
            if not blocks:
                _, falsebr = labels.pop()
            else:
                falsebr, _ = blocks[-1]
            br = ir.Branch(cond=pred, truebr=cp_label, falsebr=falsebr, loc=self.loc)
            self.current_block.append(br)