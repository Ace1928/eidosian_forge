import os
from dataclasses import dataclass, replace, field, fields
import dis
import operator
from functools import reduce
from typing import (
from collections import ChainMap
from numba_rvsdg.core.datastructures.byte_flow import ByteFlow
from numba_rvsdg.core.datastructures.scfg import SCFG
from numba_rvsdg.core.datastructures.basic_block import (
from numba_rvsdg.rendering.rendering import ByteFlowRenderer
from numba_rvsdg.core.datastructures import block_names
from numba.core.utils import MutableSortedSet, MutableSortedMap
from .regionpasses import (
class HandleConditionalPop:
    """Introduce pop-stack operations to the bytecode to correctly model
    operations that conditionally pop elements from the stack. Numba-rvsdg does
    not handle this. For example, FOR_ITER pop the stack when the iterator is
    exhausted.
    """

    def handle(self, inst: dis.Instruction) -> _ExtraBranch | None:
        fn = getattr(self, f'op_{inst.opname}', self._op_default)
        return fn(inst)

    def _op_default(self, inst: dis.Instruction) -> None:
        assert not inst.opname.endswith('OR_POP')
        return

    def op_FOR_ITER(self, inst: dis.Instruction) -> _ExtraBranch:
        br0 = ('FOR_ITER_STORE_INDVAR',)
        br1 = ('POP',)
        return _ExtraBranch((br0, br1))

    def op_JUMP_IF_TRUE_OR_POP(self, inst: dis.Instruction) -> _ExtraBranch:
        br0 = ('POP',)
        br1 = ()
        return _ExtraBranch((br0, br1))

    def op_JUMP_IF_FALSE_OR_POP(self, inst: dis.Instruction) -> _ExtraBranch:
        br0 = ('POP',)
        br1 = ()
        return _ExtraBranch((br0, br1))