from ctypes import c_uint, c_bool
from llvmlite.binding import ffi
from llvmlite.binding import passmanagers
def _populate_module_pm(self, pm):
    ffi.lib.LLVMPY_PassManagerBuilderPopulateModulePassManager(self, pm)