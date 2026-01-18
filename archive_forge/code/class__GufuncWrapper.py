from collections import namedtuple
import numpy as np
from llvmlite.ir import Constant, IRBuilder
from llvmlite import ir
from numba.core import types, cgutils
from numba.core.compiler_lock import global_compiler_lock
from numba.core.caching import make_library_cache, NullCache
class _GufuncWrapper(object):

    def __init__(self, py_func, cres, sin, sout, cache, is_parfors):
        """
        The *is_parfors* argument is a boolean that indicates if the GUfunc
        being built is to be used as a ParFors kernel. If True, it disables
        the caching on the wrapper as a separate unit because it will be linked
        into the caller function and cached along with it.
        """
        self.py_func = py_func
        self.cres = cres
        self.sin = sin
        self.sout = sout
        self.is_objectmode = self.signature.return_type == types.pyobject
        self.cache = GufWrapperCache(py_func=self.py_func) if cache else NullCache()
        self.is_parfors = bool(is_parfors)

    @property
    def library(self):
        return self.cres.library

    @property
    def context(self):
        return self.cres.target_context

    @property
    def call_conv(self):
        return self.context.call_conv

    @property
    def signature(self):
        return self.cres.signature

    @property
    def fndesc(self):
        return self.cres.fndesc

    @property
    def env(self):
        return self.cres.environment

    def _wrapper_function_type(self):
        byte_t = ir.IntType(8)
        byte_ptr_t = ir.PointerType(byte_t)
        byte_ptr_ptr_t = ir.PointerType(byte_ptr_t)
        intp_t = self.context.get_value_type(types.intp)
        intp_ptr_t = ir.PointerType(intp_t)
        fnty = ir.FunctionType(ir.VoidType(), [byte_ptr_ptr_t, intp_ptr_t, intp_ptr_t, byte_ptr_t])
        return fnty

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

    def _compile_wrapper(self, wrapper_name):
        if self.is_parfors:
            wrapperlib = self.context.codegen().create_library(str(self))
            self._build_wrapper(wrapperlib, wrapper_name)
        else:
            wrapperlib = self.cache.load_overload(self.cres.signature, self.cres.target_context)
            if wrapperlib is None:
                wrapperlib = self.context.codegen().create_library(str(self))
                wrapperlib.enable_object_caching()
                self._build_wrapper(wrapperlib, wrapper_name)
                self.cache.save_overload(self.cres.signature, wrapperlib)
        return wrapperlib

    @global_compiler_lock
    def build(self):
        wrapper_name = '__gufunc__.' + self.fndesc.mangled_name
        wrapperlib = self._compile_wrapper(wrapper_name)
        return _wrapper_info(library=wrapperlib, env=self.env, name=wrapper_name)

    def gen_loop_body(self, builder, pyapi, func, args):
        status, retval = self.call_conv.call_function(builder, func, self.signature.return_type, self.signature.args, args)
        with builder.if_then(status.is_error, likely=False):
            gil = pyapi.gil_ensure()
            self.context.call_conv.raise_error(builder, pyapi, status)
            pyapi.gil_release(gil)
        return (status.code, status.is_error)

    def gen_prologue(self, builder, pyapi):
        pass

    def gen_epilogue(self, builder, pyapi):
        pass