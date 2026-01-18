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
class CanonicalizeLoopEntry(FunctionPass):
    """A pass to canonicalize loop header by splitting it from function entry.

    This is needed for loop-lifting; esp in py3.8
    """
    _name = 'canonicalize_loop_entry'

    def __init__(self):
        FunctionPass.__init__(self)

    def run_pass(self, state):
        fir = state.func_ir
        cfg = compute_cfg_from_blocks(fir.blocks)
        status = False
        for loop in cfg.loops().values():
            if len(loop.entries) == 1:
                [entry_label] = loop.entries
                if entry_label == cfg.entry_point():
                    self._split_entry_block(fir, cfg, loop, entry_label)
                    status = True
        fir._reset_analysis_variables()
        vlt = postproc.VariableLifetime(fir.blocks)
        fir.variable_lifetime = vlt
        return status

    def _split_entry_block(self, fir, cfg, loop, entry_label):
        header_block = fir.blocks[loop.header]
        deps = set()
        for expr in header_block.find_exprs(op='iternext'):
            deps.add(expr.value)
        entry_block = fir.blocks[entry_label]
        startpt = None
        list_of_insts = list(entry_block.find_insts(ir.Assign))
        for assign in reversed(list_of_insts):
            if assign.target in deps:
                rhs = assign.value
                if isinstance(rhs, ir.Var):
                    if rhs.is_temp:
                        deps.add(rhs)
                elif isinstance(rhs, ir.Expr):
                    expr = rhs
                    if expr.op == 'getiter':
                        startpt = assign
                        if expr.value.is_temp:
                            deps.add(expr.value)
                    elif expr.op == 'call':
                        defn = guard(get_definition, fir, expr.func)
                        if isinstance(defn, ir.Global):
                            if expr.func.is_temp:
                                deps.add(expr.func)
                elif isinstance(rhs, ir.Global) and rhs.value is range:
                    startpt = assign
        if startpt is None:
            return
        splitpt = entry_block.body.index(startpt)
        new_block = entry_block.copy()
        new_block.body = new_block.body[splitpt:]
        new_block.loc = new_block.body[0].loc
        new_label = find_max_label(fir.blocks) + 1
        entry_block.body = entry_block.body[:splitpt]
        entry_block.append(ir.Jump(new_label, loc=new_block.loc))
        fir.blocks[new_label] = new_block
        fir.blocks = rename_labels(fir.blocks)