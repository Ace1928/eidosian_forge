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
def _emit_two_way_switch(self, block):
    with self.set_block(self._get_label(block.name), ir.Block(scope=self.local_scope, loc=self.loc)):
        assert set(block.branch_value_table.keys()) == {0, 1}
        cp = block.variable
        cpvar = self.local_scope.get_exact(self._get_cp_name(cp))
        truebr = self._get_label(block.branch_value_table[1])
        falsebr = self._get_label(block.branch_value_table[0])
        br = ir.Branch(cond=cpvar, truebr=truebr, falsebr=falsebr, loc=self.loc)
        self.current_block.append(br)