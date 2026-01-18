from collections import namedtuple
import numpy as np
from llvmlite.ir import Constant, IRBuilder
from llvmlite import ir
from numba.core import types, cgutils
from numba.core.compiler_lock import global_compiler_lock
from numba.core.caching import make_library_cache, NullCache
def _build_wrapper(self, library, name):
    """
        The LLVM IRBuilder code to create the gufunc wrapper.
        The *library* arg is the CodeLibrary to which the wrapper should
        be added.  The *name* arg is the name of the wrapper function being
        created.
        """
    intp_t = self.context.get_value_type(types.intp)
    fnty = self._wrapper_function_type()
    wrapper_module = library.create_ir_module('_gufunc_wrapper')
    func_type = self.call_conv.get_function_type(self.fndesc.restype, self.fndesc.argtypes)
    fname = self.fndesc.llvm_func_name
    func = ir.Function(wrapper_module, func_type, name=fname)
    func.attributes.add('alwaysinline')
    wrapper = ir.Function(wrapper_module, fnty, name)
    wrapper.linkage = 'weak_odr'
    arg_args, arg_dims, arg_steps, arg_data = wrapper.args
    arg_args.name = 'args'
    arg_dims.name = 'dims'
    arg_steps.name = 'steps'
    arg_data.name = 'data'
    builder = IRBuilder(wrapper.append_basic_block('entry'))
    loopcount = builder.load(arg_dims, name='loopcount')
    pyapi = self.context.get_python_api(builder)
    unique_syms = set()
    for grp in (self.sin, self.sout):
        for syms in grp:
            unique_syms |= set(syms)
    sym_map = {}
    for syms in self.sin:
        for s in syms:
            if s not in sym_map:
                sym_map[s] = len(sym_map)
    sym_dim = {}
    for s, i in sym_map.items():
        sym_dim[s] = builder.load(builder.gep(arg_dims, [self.context.get_constant(types.intp, i + 1)]))
    arrays = []
    step_offset = len(self.sin) + len(self.sout)
    for i, (typ, sym) in enumerate(zip(self.signature.args, self.sin + self.sout)):
        ary = GUArrayArg(self.context, builder, arg_args, arg_steps, i, step_offset, typ, sym, sym_dim)
        step_offset += len(sym)
        arrays.append(ary)
    bbreturn = builder.append_basic_block('.return')
    self.gen_prologue(builder, pyapi)
    with cgutils.for_range(builder, loopcount, intp=intp_t) as loop:
        args = [a.get_array_at_offset(loop.index) for a in arrays]
        innercall, error = self.gen_loop_body(builder, pyapi, func, args)
        cgutils.cbranch_or_continue(builder, error, bbreturn)
    builder.branch(bbreturn)
    builder.position_at_end(bbreturn)
    self.gen_epilogue(builder, pyapi)
    builder.ret_void()
    library.add_ir_module(wrapper_module)
    library.add_linking_library(self.library)