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
class InlineInlinables(FunctionPass):
    """
    This pass will inline a function wrapped by the numba.jit decorator directly
    into the site of its call depending on the value set in the 'inline' kwarg
    to the decorator.

    This is an untyped pass. CFG simplification is performed at the end of the
    pass but no block level clean up is performed on the mutated IR (typing
    information is not available to do so).
    """
    _name = 'inline_inlinables'
    _DEBUG = False

    def __init__(self):
        FunctionPass.__init__(self)

    def run_pass(self, state):
        """Run inlining of inlinables
        """
        if self._DEBUG:
            print('before inline'.center(80, '-'))
            print(state.func_ir.dump())
            print(''.center(80, '-'))
        from numba.core.inline_closurecall import InlineWorker, callee_ir_validator
        inline_worker = InlineWorker(state.typingctx, state.targetctx, state.locals, state.pipeline, state.flags, validator=callee_ir_validator)
        modified = False
        work_list = list(state.func_ir.blocks.items())
        while work_list:
            label, block = work_list.pop()
            for i, instr in enumerate(block.body):
                if isinstance(instr, ir.Assign):
                    expr = instr.value
                    if isinstance(expr, ir.Expr) and expr.op == 'call':
                        if guard(self._do_work, state, work_list, block, i, expr, inline_worker):
                            modified = True
                            break
        if modified:
            cfg = compute_cfg_from_blocks(state.func_ir.blocks)
            for dead in cfg.dead_nodes():
                del state.func_ir.blocks[dead]
            post_proc = postproc.PostProcessor(state.func_ir)
            post_proc.run()
            state.func_ir.blocks = simplify_CFG(state.func_ir.blocks)
        if self._DEBUG:
            print('after inline'.center(80, '-'))
            print(state.func_ir.dump())
            print(''.center(80, '-'))
        return True

    def _do_work(self, state, work_list, block, i, expr, inline_worker):
        from numba.core.compiler import run_frontend
        from numba.core.cpu import InlineOptions
        to_inline = None
        try:
            to_inline = state.func_ir.get_definition(expr.func)
        except Exception:
            if self._DEBUG:
                print('Cannot find definition for %s' % expr.func)
            return False
        if getattr(to_inline, 'op', False) == 'make_function':
            return False
        if getattr(to_inline, 'op', False) == 'getattr':
            val = resolve_func_from_module(state.func_ir, to_inline)
        else:
            try:
                val = getattr(to_inline, 'value', False)
            except Exception:
                raise GuardException
        if val:
            topt = getattr(val, 'targetoptions', False)
            if topt:
                inline_type = topt.get('inline', None)
                if inline_type is not None:
                    inline_opt = InlineOptions(inline_type)
                    if not inline_opt.is_never_inline:
                        do_inline = True
                        pyfunc = val.py_func
                        if inline_opt.has_cost_model:
                            py_func_ir = run_frontend(pyfunc)
                            do_inline = inline_type(expr, state.func_ir, py_func_ir)
                        if do_inline:
                            _, _, _, new_blocks = inline_worker.inline_function(state.func_ir, block, i, pyfunc)
                            if work_list is not None:
                                for blk in new_blocks:
                                    work_list.append(blk)
                            return True
        return False