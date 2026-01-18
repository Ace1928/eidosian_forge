import logging
import os
import sys
from llvmlite import ir
from llvmlite.binding import Linkage
from numba.pycc import llvm_types as lt
from numba.core.cgutils import create_constant_array
from numba.core.compiler import compile_extra, Flags
from numba.core.compiler_lock import global_compiler_lock
from numba.core.registry import cpu_target
from numba.core.runtime import nrtdynmod
from numba.core import cgutils
class ModuleCompiler(_ModuleCompiler):
    _ptr_fun = lambda ret, *args: ir.PointerType(ir.FunctionType(ret, args))
    visitproc_ty = _ptr_fun(lt._int8, lt._pyobject_head_p)
    inquiry_ty = _ptr_fun(lt._int8, lt._pyobject_head_p)
    traverseproc_ty = _ptr_fun(lt._int8, lt._pyobject_head_p, visitproc_ty, lt._void_star)
    freefunc_ty = _ptr_fun(lt._int8, lt._void_star)
    m_init_ty = _ptr_fun(lt._int8)
    _char_star = lt._int8_star
    module_def_base_ty = ir.LiteralStructType((lt._pyobject_head, m_init_ty, lt._llvm_py_ssize_t, lt._pyobject_head_p))
    module_def_ty = ir.LiteralStructType((module_def_base_ty, _char_star, _char_star, lt._llvm_py_ssize_t, _ModuleCompiler.method_def_ptr, inquiry_ty, traverseproc_ty, inquiry_ty, freefunc_ty))

    @property
    def module_create_definition(self):
        """
        Return the signature and name of the Python C API function to
        initialize the module.
        """
        signature = ir.FunctionType(lt._pyobject_head_p, (ir.PointerType(self.module_def_ty), lt._int32))
        name = 'PyModule_Create2'
        if lt._trace_refs_:
            name += 'TraceRefs'
        return (signature, name)

    @property
    def module_init_definition(self):
        """
        Return the name and signature of the module's initialization function.
        """
        signature = ir.FunctionType(lt._pyobject_head_p, ())
        return (signature, 'PyInit_' + self.module_name)

    def _emit_python_wrapper(self, llvm_module):
        create_module_fn = ir.Function(llvm_module, *self.module_create_definition)
        create_module_fn.linkage = 'external'
        mod_name_const = self.context.insert_const_string(llvm_module, self.module_name)
        mod_def_base_init = ir.Constant.literal_struct((lt._pyobject_head_init, ir.Constant(self.m_init_ty, None), ir.Constant(lt._llvm_py_ssize_t, None), ir.Constant(lt._pyobject_head_p, None)))
        mod_def_base = cgutils.add_global_variable(llvm_module, mod_def_base_init.type, '.module_def_base')
        mod_def_base.initializer = mod_def_base_init
        mod_def_base.linkage = 'internal'
        method_array = self._emit_method_array(llvm_module)
        mod_def_init = ir.Constant.literal_struct((mod_def_base_init, mod_name_const, ir.Constant(self._char_star, None), ir.Constant(lt._llvm_py_ssize_t, -1), method_array, ir.Constant(self.inquiry_ty, None), ir.Constant(self.traverseproc_ty, None), ir.Constant(self.inquiry_ty, None), ir.Constant(self.freefunc_ty, None)))
        mod_def = cgutils.add_global_variable(llvm_module, mod_def_init.type, '.module_def')
        mod_def.initializer = mod_def_init
        mod_def.linkage = 'internal'
        mod_init_fn = ir.Function(llvm_module, *self.module_init_definition)
        entry = mod_init_fn.append_basic_block('Entry')
        builder = ir.IRBuilder(entry)
        pyapi = self.context.get_python_api(builder)
        mod = builder.call(create_module_fn, (mod_def, ir.Constant(lt._int32, sys.api_version)))
        with builder.if_then(cgutils.is_null(builder, mod)):
            builder.ret(NULL.bitcast(mod_init_fn.type.pointee.return_type))
        env_array = self._emit_environment_array(llvm_module, builder, pyapi)
        envgv_array = self._emit_envgvs_array(llvm_module, builder, pyapi)
        ret = self._emit_module_init_code(llvm_module, builder, mod, method_array, env_array, envgv_array)
        if ret is not None:
            with builder.if_then(cgutils.is_not_null(builder, ret)):
                builder.ret(ir.Constant(mod.type, None))
        builder.ret(mod)
        self.dll_exports.append(mod_init_fn.name)