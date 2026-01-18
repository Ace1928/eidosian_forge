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
def _emit_environment_array(self, llvm_module, builder, pyapi):
    """
        Emit an array of env_def_t structures (see modulemixin.c)
        storing the pickled environment constants for each of the
        exported functions.
        """
    env_defs = []
    for entry in self.export_entries:
        env = self.function_environments[entry]
        env_def = pyapi.serialize_uncached(env.consts)
        env_defs.append(env_def)
    env_defs_init = create_constant_array(self.env_def_ty, env_defs)
    gv = self.context.insert_unique_const(llvm_module, '.module_environments', env_defs_init)
    return gv.gep([ZERO, ZERO])