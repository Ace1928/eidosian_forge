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
class IRProcessing(FunctionPass):
    _name = 'ir_processing'

    def __init__(self):
        FunctionPass.__init__(self)

    def run_pass(self, state):
        func_ir = state['func_ir']
        post_proc = postproc.PostProcessor(func_ir)
        post_proc.run()
        if config.DEBUG or config.DUMP_IR:
            name = func_ir.func_id.func_qualname
            print(('IR DUMP: %s' % name).center(80, '-'))
            func_ir.dump()
            if func_ir.is_generator:
                print(('GENERATOR INFO: %s' % name).center(80, '-'))
                func_ir.dump_generator_info()
        return True