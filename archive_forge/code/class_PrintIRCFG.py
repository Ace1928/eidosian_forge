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
@register_pass(mutates_CFG=False, analysis_only=True)
class PrintIRCFG(FunctionPass):
    _name = 'print_ir_cfg'

    def __init__(self):
        FunctionPass.__init__(self)
        self._ver = 0

    def run_pass(self, state):
        fir = state.func_ir
        self._ver += 1
        fir.render_dot(filename_prefix='v{}'.format(self._ver)).render()
        return False