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