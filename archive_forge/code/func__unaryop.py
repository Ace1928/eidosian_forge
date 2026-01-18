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
def _unaryop(self, opname: str, inst: dis.Instruction):
    op = Op(opname=opname, bc_inst=inst)
    op.add_input('val', self.pop())
    self.push(op.add_output('out'))