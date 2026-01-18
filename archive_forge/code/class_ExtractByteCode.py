from collections import defaultdict, namedtuple
from contextlib import contextmanager
from copy import deepcopy, copy
import warnings
from numba.core.compiler_machinery import (FunctionPass, AnalysisPass,
from numba.core import (errors, types, ir, bytecode, postproc, rewrites, config,
from numba.misc.special import literal_unroll
from numba.core.analysis import (dead_branch_prune, rewrite_semantic_constants,
from numba.core.ir_utils import (guard, resolve_func_from_module, simplify_CFG,
from numba.core.ssa import reconstruct_ssa
from numba.core import interpreter
@register_pass(mutates_CFG=True, analysis_only=False)
class ExtractByteCode(FunctionPass):
    _name = 'extract_bytecode'

    def __init__(self):
        FunctionPass.__init__(self)

    def run_pass(self, state):
        """
        Extract bytecode from function
        """
        func_id = state['func_id']
        bc = bytecode.ByteCode(func_id)
        if config.DUMP_BYTECODE:
            print(bc.dump())
        state['bc'] = bc
        return True