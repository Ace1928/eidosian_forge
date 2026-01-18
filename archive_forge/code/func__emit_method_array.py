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
def _emit_method_array(self, llvm_module):
    """
        Collect exported methods and emit a PyMethodDef array.

        :returns: a pointer to the PyMethodDef array.
        """
    method_defs = []
    for entry in self.export_entries:
        name = entry.symbol
        llvm_func_name = self._mangle_method_symbol(name)
        fnty = self.exported_function_types[entry]
        lfunc = ir.Function(llvm_module, fnty, llvm_func_name)
        method_name = self.context.insert_const_string(llvm_module, name)
        method_def_const = ir.Constant.literal_struct((method_name, ir.Constant.bitcast(lfunc, lt._void_star), METH_VARARGS_AND_KEYWORDS, NULL))
        method_defs.append(method_def_const)
    sentinel = ir.Constant.literal_struct([NULL, NULL, ZERO, NULL])
    method_defs.append(sentinel)
    method_array_init = create_constant_array(self.method_def_ty, method_defs)
    method_array = cgutils.add_global_variable(llvm_module, method_array_init.type, '.module_methods')
    method_array.initializer = method_array_init
    method_array.linkage = 'internal'
    method_array_ptr = ir.Constant.gep(method_array, [ZERO, ZERO])
    return method_array_ptr