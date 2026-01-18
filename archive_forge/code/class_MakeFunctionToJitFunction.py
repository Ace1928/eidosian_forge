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
class MakeFunctionToJitFunction(FunctionPass):
    """
    This swaps an ir.Expr.op == "make_function" i.e. a closure, for a compiled
    function containing the closure body and puts it in ir.Global. It's a 1:1
    statement value swap. `make_function` is already untyped
    """
    _name = 'make_function_op_code_to_jit_function'

    def __init__(self):
        FunctionPass.__init__(self)

    def run_pass(self, state):
        from numba import njit
        func_ir = state.func_ir
        mutated = False
        for idx, blk in func_ir.blocks.items():
            for stmt in blk.body:
                if isinstance(stmt, ir.Assign):
                    if isinstance(stmt.value, ir.Expr):
                        if stmt.value.op == 'make_function':
                            node = stmt.value
                            getdef = func_ir.get_definition
                            kw_default = getdef(node.defaults)
                            ok = False
                            if kw_default is None or isinstance(kw_default, ir.Const):
                                ok = True
                            elif isinstance(kw_default, tuple):
                                ok = all([isinstance(getdef(x), ir.Const) for x in kw_default])
                            elif isinstance(kw_default, ir.Expr):
                                if kw_default.op != 'build_tuple':
                                    continue
                                ok = all([isinstance(getdef(x), ir.Const) for x in kw_default.items])
                            if not ok:
                                continue
                            pyfunc = convert_code_obj_to_function(node, func_ir)
                            func = njit()(pyfunc)
                            new_node = ir.Global(node.code.co_name, func, stmt.loc)
                            stmt.value = new_node
                            mutated |= True
        if mutated:
            post_proc = postproc.PostProcessor(func_ir)
            post_proc.run()
        return mutated