import numpy
import operator
from numba.core import types, ir, config, cgutils, errors
from numba.core.ir_utils import (
from numba.core.analysis import compute_cfg_from_blocks
from numba.core.typing import npydecl, signature
import copy
from numba.core.extending import intrinsic
import llvmlite
def _combine_to_new_block(self, block, pending_transforms):
    """Combine the new instructions from previous pass into a new block
        body.
        """
    new_body = []
    for inst, pre, post in pending_transforms:
        for instr in pre:
            new_body.append(instr)
        new_body.append(inst)
        for instr in post:
            new_body.append(instr)
    block.body = new_body