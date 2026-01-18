import types as pytypes  # avoid confusion with numba.types
import copy
import ctypes
import numba.core.analysis
from numba.core import types, typing, errors, ir, rewrites, config, ir_utils
from numba.parfors.parfor import internal_prange
from numba.core.ir_utils import (
from numba.core.analysis import (
from numba.core import postproc
from numba.np.unsafe.ndarray import empty_inferred as unsafe_empty_inferred
import numpy as np
import operator
import numba.misc.special
class InlineWorker(object):
    """ A worker class for inlining, this is a more advanced version of
    `inline_closure_call` in that it permits inlining from function type, Numba
    IR and code object. It also, runs the entire untyped compiler pipeline on
    the inlinee to ensure that it is transformed as though it were compiled
    directly.
    """

    def __init__(self, typingctx=None, targetctx=None, locals=None, pipeline=None, flags=None, validator=callee_ir_validator, typemap=None, calltypes=None):
        """
        Instantiate a new InlineWorker, all arguments are optional though some
        must be supplied together for certain use cases. The methods will refuse
        to run if the object isn't configured in the manner needed. Args are the
        same as those in a numba.core.Compiler.state, except the validator which
        is a function taking Numba IR and validating it for use when inlining
        (this is optional and really to just provide better error messages about
        things which the inliner cannot handle like yield in closure).
        """

        def check(arg, name):
            if arg is None:
                raise TypeError('{} must not be None'.format(name))
        from numba.core.compiler import DefaultPassBuilder
        compiler_args = (targetctx, locals, pipeline, flags)
        compiler_group = [x is not None for x in compiler_args]
        if any(compiler_group) and (not all(compiler_group)):
            check(targetctx, 'targetctx')
            check(locals, 'locals')
            check(pipeline, 'pipeline')
            check(flags, 'flags')
        elif all(compiler_group):
            check(typingctx, 'typingctx')
        self._compiler_pipeline = DefaultPassBuilder.define_untyped_pipeline
        self.typingctx = typingctx
        self.targetctx = targetctx
        self.locals = locals
        self.pipeline = pipeline
        self.flags = flags
        self.validator = validator
        self.debug_print = _make_debug_print('InlineWorker')
        pair = (typemap, calltypes)
        pair_is_none = [x is None for x in pair]
        if any(pair_is_none) and (not all(pair_is_none)):
            msg = 'typemap and calltypes must both be either None or have a value, got: %s, %s'
            raise TypeError(msg % pair)
        self._permit_update_type_and_call_maps = not all(pair_is_none)
        self.typemap = typemap
        self.calltypes = calltypes

    def inline_ir(self, caller_ir, block, i, callee_ir, callee_freevars, arg_typs=None):
        """ Inlines the callee_ir in the caller_ir at statement index i of block
        `block`, callee_freevars are the free variables for the callee_ir. If
        the callee_ir is derived from a function `func` then this is
        `func.__code__.co_freevars`. If `arg_typs` is given and the InlineWorker
        instance was initialized with a typemap and calltypes then they will be
        appropriately updated based on the arg_typs.
        """

        def copy_ir(the_ir):
            kernel_copy = the_ir.copy()
            kernel_copy.blocks = {}
            for block_label, block in the_ir.blocks.items():
                new_block = copy.deepcopy(the_ir.blocks[block_label])
                kernel_copy.blocks[block_label] = new_block
            return kernel_copy
        callee_ir = copy_ir(callee_ir)
        if self.validator is not None:
            self.validator(callee_ir)
        callee_ir_original = copy_ir(callee_ir)
        scope = block.scope
        instr = block.body[i]
        call_expr = instr.value
        callee_blocks = callee_ir.blocks
        max_label = max(ir_utils._the_max_label.next(), max(caller_ir.blocks.keys()))
        callee_blocks = add_offset_to_labels(callee_blocks, max_label + 1)
        callee_blocks = simplify_CFG(callee_blocks)
        callee_ir.blocks = callee_blocks
        min_label = min(callee_blocks.keys())
        max_label = max(callee_blocks.keys())
        ir_utils._the_max_label.update(max_label)
        self.debug_print('After relabel')
        _debug_dump(callee_ir)
        callee_scopes = _get_all_scopes(callee_blocks)
        self.debug_print('callee_scopes = ', callee_scopes)
        assert len(callee_scopes) == 1
        callee_scope = callee_scopes[0]
        var_dict = {}
        for var in tuple(callee_scope.localvars._con.values()):
            if not var.name in callee_freevars:
                inlined_name = _created_inlined_var_name(callee_ir.func_id.unique_name, var.name)
                new_var = scope.redefine(inlined_name, loc=var.loc)
                callee_scope.redefine(inlined_name, loc=var.loc)
                var_dict[var.name] = new_var
        self.debug_print('var_dict = ', var_dict)
        replace_vars(callee_blocks, var_dict)
        self.debug_print('After local var rename')
        _debug_dump(callee_ir)
        callee_func = callee_ir.func_id.func
        args = _get_callee_args(call_expr, callee_func, block.body[i].loc, caller_ir)
        if self._permit_update_type_and_call_maps:
            if arg_typs is None:
                raise TypeError('arg_typs should have a value not None')
            self.update_type_and_call_maps(callee_ir, arg_typs)
            callee_blocks = callee_ir.blocks
        self.debug_print('After arguments rename: ')
        _debug_dump(callee_ir)
        _replace_args_with(callee_blocks, args)
        new_blocks = []
        new_block = ir.Block(scope, block.loc)
        new_block.body = block.body[i + 1:]
        new_label = next_label()
        caller_ir.blocks[new_label] = new_block
        new_blocks.append((new_label, new_block))
        block.body = block.body[:i]
        block.body.append(ir.Jump(min_label, instr.loc))
        topo_order = find_topo_order(callee_blocks)
        _replace_returns(callee_blocks, instr.target, new_label)
        if instr.target.name in caller_ir._definitions and call_expr in caller_ir._definitions[instr.target.name]:
            caller_ir._definitions[instr.target.name].remove(call_expr)
        for label in topo_order:
            block = callee_blocks[label]
            block.scope = scope
            _add_definitions(caller_ir, block)
            caller_ir.blocks[label] = block
            new_blocks.append((label, block))
        self.debug_print('After merge in')
        _debug_dump(caller_ir)
        return (callee_ir_original, callee_blocks, var_dict, new_blocks)

    def inline_function(self, caller_ir, block, i, function, arg_typs=None):
        """ Inlines the function in the caller_ir at statement index i of block
        `block`. If `arg_typs` is given and the InlineWorker instance was
        initialized with a typemap and calltypes then they will be appropriately
        updated based on the arg_typs.
        """
        callee_ir = self.run_untyped_passes(function)
        freevars = function.__code__.co_freevars
        return self.inline_ir(caller_ir, block, i, callee_ir, freevars, arg_typs=arg_typs)

    def run_untyped_passes(self, func, enable_ssa=False):
        """
        Run the compiler frontend's untyped passes over the given Python
        function, and return the function's canonical Numba IR.

        Disable SSA transformation by default, since the call site won't be in
        SSA form and self.inline_ir depends on this being the case.
        """
        from numba.core.compiler import StateDict, _CompileStatus
        from numba.core.untyped_passes import ExtractByteCode
        from numba.core import bytecode
        from numba.parfors.parfor import ParforDiagnostics
        state = StateDict()
        state.func_ir = None
        state.typingctx = self.typingctx
        state.targetctx = self.targetctx
        state.locals = self.locals
        state.pipeline = self.pipeline
        state.flags = self.flags
        state.flags.enable_ssa = enable_ssa
        state.func_id = bytecode.FunctionIdentity.from_function(func)
        state.typemap = None
        state.calltypes = None
        state.type_annotation = None
        state.status = _CompileStatus(False)
        state.return_type = None
        state.parfor_diagnostics = ParforDiagnostics()
        state.metadata = {}
        ExtractByteCode().run_pass(state)
        state.args = len(state.bc.func_id.pysig.parameters) * (types.pyobject,)
        pm = self._compiler_pipeline(state)
        pm.finalize()
        pm.run(state)
        return state.func_ir

    def update_type_and_call_maps(self, callee_ir, arg_typs):
        """ Updates the type and call maps based on calling callee_ir with
        arguments from arg_typs"""
        from numba.core.ssa import reconstruct_ssa
        from numba.core.typed_passes import PreLowerStripPhis
        if not self._permit_update_type_and_call_maps:
            msg = 'InlineWorker instance not configured correctly, typemap or calltypes missing in initialization.'
            raise ValueError(msg)
        from numba.core import typed_passes
        callee_ir._definitions = ir_utils.build_definitions(callee_ir.blocks)
        numba.core.analysis.dead_branch_prune(callee_ir, arg_typs)
        callee_ir = reconstruct_ssa(callee_ir)
        callee_ir._definitions = ir_utils.build_definitions(callee_ir.blocks)
        [f_typemap, _f_return_type, f_calltypes, _] = typed_passes.type_inference_stage(self.typingctx, self.targetctx, callee_ir, arg_typs, None)
        callee_ir = PreLowerStripPhis()._strip_phi_nodes(callee_ir)
        callee_ir._definitions = ir_utils.build_definitions(callee_ir.blocks)
        canonicalize_array_math(callee_ir, f_typemap, f_calltypes, self.typingctx)
        arg_names = [vname for vname in f_typemap if vname.startswith('arg.')]
        for a in arg_names:
            f_typemap.pop(a)
        self.typemap.update(f_typemap)
        self.calltypes.update(f_calltypes)