import unittest
from unittest.case import TestCase
import warnings
import numpy as np
from numba import objmode
from numba.core import ir, compiler
from numba.core import errors
from numba.core.compiler import (
from numba.core.compiler_machinery import (
from numba.core.untyped_passes import (
from numba import njit
@register_pass(mutates_CFG=False, analysis_only=False)
class RemoveVarInScope(FunctionPass):
    _name = '_remove_var_in_scope'

    def __init__(self):
        FunctionPass.__init__(self)

    def run_pass(self, state):
        func_ir = state.func_ir
        for blk in func_ir.blocks.values():
            oldscope = blk.scope
            blk.scope = ir.Scope(parent=oldscope.parent, loc=oldscope.loc)
        return True