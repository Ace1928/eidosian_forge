import sys
import copy
import logging
import numpy as np
from numba import njit, jit, types
from numba.core import errors, ir
from numba.core.compiler_machinery import FunctionPass, register_pass
from numba.core.compiler import DefaultPassBuilder, CompilerBase
from numba.core.untyped_passes import ReconstructSSA, PreserveIR
from numba.core.typed_passes import NativeLowering
from numba.extending import overload
from numba.tests.support import MemoryLeakMixin, TestCase, override_config
@register_pass(mutates_CFG=False, analysis_only=True)
class CheckSSAMinimal(FunctionPass):
    _name = self.__class__.__qualname__ + '.CheckSSAMinimal'

    def __init__(self):
        super().__init__(self)

    def run_pass(self, state):
        ct = 0
        for blk in state.func_ir.blocks.values():
            ct += len(list(blk.find_exprs('phi')))
        phi_counter.append(ct)
        return True