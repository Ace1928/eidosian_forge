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