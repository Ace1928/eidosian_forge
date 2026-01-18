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
def _converter_to_ddgblock(block, converter) -> DDGBlock:
    blk = DDGBlock(name=block.name, _jump_targets=block._jump_targets, backedges=block.backedges, in_effect=converter.in_effect, out_effect=converter.effect, in_stackvars=list(converter.incoming_stackvars), out_stackvars=list(converter.stack), in_vars=MutableSortedMap(converter.incoming_vars), out_vars=MutableSortedMap(converter.varmap))
    return blk