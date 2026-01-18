import os
from . import sysconfig
from .errors import CompileError, DistutilsExecError
from .unixccompiler import UnixCCompiler
def _get_zos_compiler_name(self):
    zos_compiler_names = [os.path.basename(binary) for envvar in ('CC', 'CXX', 'LDSHARED') if (binary := os.environ.get(envvar, None))]
    if len(zos_compiler_names) == 0:
        return 'ibm-openxl'
    zos_compilers = {}
    for compiler in ('ibm-clang', 'ibm-clang64', 'ibm-clang++', 'ibm-clang++64', 'clang', 'clang++', 'clang-14'):
        zos_compilers[compiler] = 'ibm-openxl'
    for compiler in ('xlclang', 'xlclang++', 'njsc', 'njsc++'):
        zos_compilers[compiler] = 'ibm-xlclang'
    for compiler in ('xlc', 'xlC', 'xlc++'):
        zos_compilers[compiler] = 'ibm-xlc'
    return zos_compilers.get(zos_compiler_names[0], 'ibm-openxl')