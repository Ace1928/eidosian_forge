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
def _insert_outgoing_phis(self):
    """
        Add assignments to forward requested outgoing values
        to subsequent blocks.
        """
    for phiname, varname in self.dfainfo.outgoing_phis.items():
        target = self.current_scope.get_or_define(phiname, loc=self.loc)
        try:
            val = self.get(varname)
        except ir.NotDefinedError:
            assert PYVERSION in ((3, 11), (3, 12)), 'unexpected missing definition'
            val = ir.Const(value=None, loc=self.loc)
        stmt = ir.Assign(value=val, target=target, loc=self.loc)
        self.definitions[target.name].append(stmt.value)
        if not self.current_block.is_terminated:
            self.current_block.append(stmt)
        else:
            self.current_block.insert_before_terminator(stmt)