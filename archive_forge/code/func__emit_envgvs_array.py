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
def _emit_envgvs_array(self, llvm_module, builder, pyapi):
    """
        Emit an array of Environment pointers that needs to be filled at
        initialization.
        """
    env_setters = []
    for entry in self.export_entries:
        envgv_name = self.environment_gvs[entry]
        gv = self.context.declare_env_global(llvm_module, envgv_name)
        envgv = gv.bitcast(lt._void_star)
        env_setters.append(envgv)
    env_setters_init = create_constant_array(lt._void_star, env_setters)
    gv = self.context.insert_unique_const(llvm_module, '.module_envgvs', env_setters_init)
    return gv.gep([ZERO, ZERO])