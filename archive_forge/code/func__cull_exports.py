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
@global_compiler_lock
def _cull_exports(self):
    """Read all the exported functions/modules in the translator
        environment, and join them into a single LLVM module.
        """
    self.exported_function_types = {}
    self.function_environments = {}
    self.environment_gvs = {}
    codegen = self.context.codegen()
    library = codegen.create_library(self.module_name)
    flags = Flags()
    flags.no_compile = True
    if not self.export_python_wrap:
        flags.no_cpython_wrapper = True
        flags.no_cfunc_wrapper = True
    if self.use_nrt:
        flags.nrt = True
        nrt_module, _ = nrtdynmod.create_nrt_module(self.context)
        library.add_ir_module(nrt_module)
    for entry in self.export_entries:
        cres = compile_extra(self.typing_context, self.context, entry.function, entry.signature.args, entry.signature.return_type, flags, locals={}, library=library)
        func_name = cres.fndesc.llvm_func_name
        llvm_func = cres.library.get_function(func_name)
        if self.export_python_wrap:
            llvm_func.linkage = 'internal'
            wrappername = cres.fndesc.llvm_cpython_wrapper_name
            wrapper = cres.library.get_function(wrappername)
            wrapper.name = self._mangle_method_symbol(entry.symbol)
            wrapper.linkage = 'external'
            fnty = cres.target_context.call_conv.get_function_type(cres.fndesc.restype, cres.fndesc.argtypes)
            self.exported_function_types[entry] = fnty
            self.function_environments[entry] = cres.environment
            self.environment_gvs[entry] = cres.fndesc.env_name
        else:
            llvm_func.name = entry.symbol
            self.dll_exports.append(entry.symbol)
    if self.export_python_wrap:
        wrapper_module = library.create_ir_module('wrapper')
        self._emit_python_wrapper(wrapper_module)
        library.add_ir_module(wrapper_module)
    library.finalize()
    for fn in library.get_defined_functions():
        if fn.name not in self.dll_exports:
            if fn.linkage in {Linkage.private, Linkage.internal}:
                fn.visibility = 'default'
            else:
                fn.visibility = 'hidden'
    return library