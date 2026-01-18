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
@register_pass(mutates_CFG=False, analysis_only=False)
class PreserveIR(AnalysisPass):
    """
    Preserves the IR in the metadata
    """
    _name = 'preserve_ir'

    def __init__(self):
        AnalysisPass.__init__(self)

    def run_pass(self, state):
        state.metadata['preserved_ir'] = state.func_ir.copy()
        return False