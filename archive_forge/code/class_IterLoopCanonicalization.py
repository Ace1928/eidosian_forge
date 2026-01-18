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
class IterLoopCanonicalization(FunctionPass):
    """ Transforms loops that are induced by `getiter` into range() driven loops
    If the typemap is available this will only impact Tuple and UniTuple, if it
    is not available it will impact all matching loops.
    """
    _name = 'iter_loop_canonicalisation'
    _DEBUG = False
    _accepted_types = (types.BaseTuple, types.LiteralList)
    _accepted_calls = (literal_unroll,)

    def __init__(self):
        FunctionPass.__init__(self)

    def assess_loop(self, loop, func_ir, partial_typemap=None):
        iternexts = [_ for _ in func_ir.blocks[loop.header].find_exprs('iternext')]
        if len(iternexts) != 1:
            return False
        for iternext in iternexts:
            phi = guard(get_definition, func_ir, iternext.value)
            if phi is None:
                return False
            if getattr(phi, 'op', False) == 'getiter':
                if partial_typemap:
                    phi_val_defn = guard(get_definition, func_ir, phi.value)
                    if not isinstance(phi_val_defn, ir.Expr):
                        return False
                    if not phi_val_defn.op == 'call':
                        return False
                    call = guard(get_definition, func_ir, phi_val_defn)
                    if call is None or len(call.args) != 1:
                        return False
                    func_var = guard(get_definition, func_ir, call.func)
                    func = guard(get_definition, func_ir, func_var)
                    if func is None or not isinstance(func, (ir.Global, ir.FreeVar)):
                        return False
                    if func.value is None or func.value not in self._accepted_calls:
                        return False
                    ty = partial_typemap.get(call.args[0].name, None)
                    if ty and isinstance(ty, self._accepted_types):
                        return len(loop.entries) == 1
                else:
                    return len(loop.entries) == 1

    def transform(self, loop, func_ir, cfg):

        def get_range(a):
            return range(len(a))
        iternext = [_ for _ in func_ir.blocks[loop.header].find_exprs('iternext')][0]
        LOC = func_ir.blocks[loop.header].loc
        scope = func_ir.blocks[loop.header].scope
        get_range_var = scope.redefine('CANONICALISER_get_range_gbl', LOC)
        get_range_global = ir.Global('get_range', get_range, LOC)
        assgn = ir.Assign(get_range_global, get_range_var, LOC)
        loop_entry = tuple(loop.entries)[0]
        entry_block = func_ir.blocks[loop_entry]
        entry_block.body.insert(0, assgn)
        iterarg = guard(get_definition, func_ir, iternext.value)
        if iterarg is not None:
            iterarg = iterarg.value
        idx = 0
        for stmt in entry_block.body:
            if isinstance(stmt, ir.Assign):
                if isinstance(stmt.value, ir.Expr) and stmt.value.op == 'getiter':
                    break
            idx += 1
        else:
            raise ValueError('problem')
        call_get_range_var = scope.redefine('CANONICALISER_call_get_range', LOC)
        make_call = ir.Expr.call(get_range_var, (stmt.value.value,), (), LOC)
        assgn_call = ir.Assign(make_call, call_get_range_var, LOC)
        entry_block.body.insert(idx, assgn_call)
        entry_block.body[idx + 1].value.value = call_get_range_var
        glbls = copy(func_ir.func_id.func.__globals__)
        from numba.core.inline_closurecall import inline_closure_call
        inline_closure_call(func_ir, glbls, entry_block, idx, get_range)
        kill = entry_block.body.index(assgn)
        entry_block.body.pop(kill)
        induction_vars = set()
        header_block = func_ir.blocks[loop.header]
        ind = [x for x in header_block.find_exprs('pair_first')]
        for x in ind:
            induction_vars.add(func_ir.get_assignee(x, loop.header))
        tmp = set()
        for x in induction_vars:
            try:
                tmp.add(func_ir.get_assignee(x, loop.header))
            except ValueError:
                pass
        induction_vars |= tmp
        induction_var_names = set([x.name for x in induction_vars])
        succ = set()
        for lbl in loop.exits:
            succ |= set([x[0] for x in cfg.successors(lbl)])
        check_blocks = (loop.body | loop.exits | succ) ^ {loop.header}
        for lbl in check_blocks:
            for stmt in func_ir.blocks[lbl].body:
                if isinstance(stmt, ir.Assign):
                    try:
                        lookup = getattr(stmt.value, 'name', None)
                    except KeyError:
                        continue
                    if lookup and lookup in induction_var_names:
                        stmt.value = ir.Expr.getitem(iterarg, stmt.value, stmt.loc)
        post_proc = postproc.PostProcessor(func_ir)
        post_proc.run()

    def run_pass(self, state):
        func_ir = state.func_ir
        cfg = compute_cfg_from_blocks(func_ir.blocks)
        loops = cfg.loops()
        mutated = False
        for header, loop in loops.items():
            stat = self.assess_loop(loop, func_ir, state.typemap)
            if stat:
                if self._DEBUG:
                    print('Canonicalising loop', loop)
                self.transform(loop, func_ir, cfg)
                mutated = True
            elif self._DEBUG:
                print('NOT Canonicalising loop', loop)
        func_ir.blocks = simplify_CFG(func_ir.blocks)
        return mutated