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
class InlineClosureLikes(FunctionPass):
    _name = 'inline_closure_likes'

    def __init__(self):
        FunctionPass.__init__(self)

    def run_pass(self, state):
        assert state.func_ir
        typed_pass = not isinstance(state.return_type, types.misc.PyObject)
        from numba.core.inline_closurecall import InlineClosureCallPass
        inline_pass = InlineClosureCallPass(state.func_ir, state.flags.auto_parallel, state.parfor_diagnostics.replaced_fns, typed_pass)
        inline_pass.run()
        post_proc = postproc.PostProcessor(state.func_ir)
        post_proc.run()
        fixup_var_define_in_scope(state.func_ir.blocks)
        return True