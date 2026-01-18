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
@register_pass(analysis_only=False, mutates_CFG=True)
class CloneFoobarAssignments(FunctionPass):
    _name = 'clone_foobar_assignments_pass'

    def __init__(self):
        FunctionPass.__init__(self)

    def run_pass(self, state):
        mutated = False
        for blk in state.func_ir.blocks.values():
            to_clone = []
            for assign in blk.find_insts(ir.Assign):
                if assign.target.name == 'foobar':
                    to_clone.append(assign)
            for assign in to_clone:
                clone = copy.deepcopy(assign)
                blk.insert_after(clone, assign)
                mutated = True
                cloned.append(clone)
        return mutated