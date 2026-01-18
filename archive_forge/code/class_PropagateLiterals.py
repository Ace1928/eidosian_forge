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
class PropagateLiterals(FunctionPass):
    """Implement literal propagation based on partial type inference"""
    _name = 'PropagateLiterals'

    def __init__(self):
        FunctionPass.__init__(self)

    def get_analysis_usage(self, AU):
        AU.add_required(ReconstructSSA)

    def run_pass(self, state):
        func_ir = state.func_ir
        typemap = state.typemap
        flags = state.flags
        accepted_functions = ('isinstance', 'hasattr')
        if not hasattr(func_ir, '_definitions') and (not flags.enable_ssa):
            func_ir._definitions = build_definitions(func_ir.blocks)
        changed = False
        for block in func_ir.blocks.values():
            for assign in block.find_insts(ir.Assign):
                value = assign.value
                if isinstance(value, (ir.Arg, ir.Const, ir.FreeVar, ir.Global)):
                    continue
                if isinstance(value, ir.Expr) and value.op in ('cast', 'build_map', 'build_list', 'build_tuple', 'build_set'):
                    continue
                target = assign.target
                if not flags.enable_ssa:
                    if guard(get_definition, func_ir, target.name) is None:
                        continue
                if isinstance(value, ir.Expr) and value.op == 'call':
                    fn = guard(get_definition, func_ir, value.func.name)
                    if fn is None:
                        continue
                    if not (isinstance(fn, ir.Global) and fn.name in accepted_functions):
                        continue
                    for arg in value.args:
                        iv = func_ir._definitions[arg.name]
                        assert len(iv) == 1
                        if isinstance(iv[0], ir.Expr) and iv[0].op == 'phi':
                            msg = f'{fn.name}() cannot determine the type of variable "{arg.unversioned_name}" due to a branch.'
                            raise errors.NumbaTypeError(msg, loc=assign.loc)
                if isinstance(value, ir.Expr) and value.op == 'phi':
                    v = [typemap.get(inc.name) for inc in value.incoming_values]
                    if v[0] is not None and any([v[0] != vi for vi in v]):
                        continue
                lit = typemap.get(target.name, None)
                if lit and isinstance(lit, types.Literal):
                    rhs = ir.Const(lit.literal_value, assign.loc)
                    new_assign = ir.Assign(rhs, target, assign.loc)
                    block.insert_after(new_assign, assign)
                    block.remove(assign)
                    changed = True
        state.typemap = None
        state.calltypes = None
        if changed:
            func_ir._definitions = build_definitions(func_ir.blocks)
        return changed