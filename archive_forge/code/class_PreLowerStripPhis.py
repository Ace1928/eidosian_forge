import abc
from contextlib import contextmanager
from collections import defaultdict, namedtuple
from functools import partial
from copy import copy
import warnings
from numba.core import (errors, types, typing, ir, funcdesc, rewrites,
from numba.parfors.parfor import PreParforPass as _parfor_PreParforPass
from numba.parfors.parfor import ParforPass as _parfor_ParforPass
from numba.parfors.parfor import ParforFusionPass as _parfor_ParforFusionPass
from numba.parfors.parfor import ParforPreLoweringPass as \
from numba.parfors.parfor import Parfor
from numba.parfors.parfor_lowering import ParforLower
from numba.core.compiler_machinery import (FunctionPass, LoweringPass,
from numba.core.annotations import type_annotations
from numba.core.ir_utils import (raise_on_unsupported_feature, warn_deprecated,
from numba.core import postproc
from llvmlite import binding as llvm
@register_pass(mutates_CFG=False, analysis_only=False)
class PreLowerStripPhis(FunctionPass):
    """Remove phi nodes (ir.Expr.phi) introduced by SSA.

    This is needed before Lowering because the phi nodes in Numba IR do not
    match the semantics of phi nodes in LLVM IR. In Numba IR, phi nodes may
    expand into multiple LLVM instructions.
    """
    _name = 'strip_phis'

    def __init__(self):
        FunctionPass.__init__(self)

    def run_pass(self, state):
        state.func_ir = self._strip_phi_nodes(state.func_ir)
        state.func_ir._definitions = build_definitions(state.func_ir.blocks)
        if 'flags' in state and state.flags.auto_parallel.enabled:
            self._simplify_conditionally_defined_variable(state.func_ir)
            state.func_ir._definitions = build_definitions(state.func_ir.blocks)
        post_proc = postproc.PostProcessor(state.func_ir)
        post_proc.run(emit_dels=False)
        if state.func_ir.generator_info is not None and state.typemap is not None:
            gentype = state.return_type
            state_vars = state.func_ir.generator_info.state_vars
            state_types = [state.typemap[k] for k in state_vars]
            state.return_type = types.Generator(gen_func=gentype.gen_func, yield_type=gentype.yield_type, arg_types=gentype.arg_types, state_types=state_types, has_finalizer=gentype.has_finalizer)
        return True

    def _strip_phi_nodes(self, func_ir):
        """Strip Phi nodes from ``func_ir``

        For each phi node, put incoming value to their respective incoming
        basic-block at possibly the latest position (i.e. after the latest
        assignment to the corresponding variable).
        """
        exporters = defaultdict(list)
        phis = set()
        for label, block in func_ir.blocks.items():
            for assign in block.find_insts(ir.Assign):
                if isinstance(assign.value, ir.Expr):
                    if assign.value.op == 'phi':
                        phis.add(assign)
                        phi = assign.value
                        for ib, iv in zip(phi.incoming_blocks, phi.incoming_values):
                            exporters[ib].append((assign.target, iv))
        newblocks = {}
        for label, block in func_ir.blocks.items():
            newblk = copy(block)
            newblocks[label] = newblk
            newblk.body = [stmt for stmt in block.body if stmt not in phis]
            for target, rhs in exporters[label]:
                if rhs is ir.UNDEFINED:
                    rhs = ir.Expr.null(loc=func_ir.loc)
                assign = ir.Assign(target=target, value=rhs, loc=rhs.loc)
                assignments = [stmt for stmt in newblk.find_insts(ir.Assign) if stmt.target == rhs]
                if assignments:
                    last_assignment = assignments[-1]
                    newblk.insert_after(assign, last_assignment)
                else:
                    newblk.prepend(assign)
        func_ir.blocks = newblocks
        return func_ir

    def _simplify_conditionally_defined_variable(self, func_ir):
        """
        Rewrite assignments like:

            ver1 = null()
            ...
            ver1 = ver
            ...
            uses(ver1)

        into:
            # delete all assignments to ver1
            uses(ver)

        This is only needed for parfors because the SSA pass will create extra
        variable assignments that the parfor code does not expect.
        This pass helps avoid problems by reverting the effect of SSA.
        """
        any_block = next(iter(func_ir.blocks.values()))
        scope = any_block.scope
        defs = func_ir._definitions

        def unver_or_undef(unver, defn):
            if isinstance(defn, ir.Var):
                if defn.unversioned_name == unver:
                    return True
            elif isinstance(defn, ir.Expr):
                if defn.op == 'null':
                    return True
            return False

        def legalize_all_versioned_names(var):
            if not var.versioned_names:
                return False
            for versioned in var.versioned_names:
                vs = defs.get(versioned, ())
                if not all(map(partial(unver_or_undef, k), vs)):
                    return False
            return True
        suspects = set()
        for k in defs:
            try:
                var = scope.get_exact(k)
            except errors.NotDefinedError:
                continue
            if var.unversioned_name == k:
                if legalize_all_versioned_names(var):
                    suspects.add(var)
        delete_set = set()
        replace_map = {}
        for var in suspects:
            for versioned in var.versioned_names:
                ver_var = scope.get_exact(versioned)
                delete_set.add(ver_var)
                replace_map[versioned] = var
        for _label, blk in func_ir.blocks.items():
            for assign in blk.find_insts(ir.Assign):
                if assign.target in delete_set:
                    blk.remove(assign)
        replace_vars(func_ir.blocks, replace_map)