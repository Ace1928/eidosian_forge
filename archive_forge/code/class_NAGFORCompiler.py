import sys
import re
from numpy.distutils.fcompiler import FCompiler
class NAGFORCompiler(BaseNAGFCompiler):
    compiler_type = 'nagfor'
    description = 'NAG Fortran Compiler'
    executables = {'version_cmd': ['nagfor', '-V'], 'compiler_f77': ['nagfor', '-fixed'], 'compiler_fix': ['nagfor', '-fixed'], 'compiler_f90': ['nagfor'], 'linker_so': ['nagfor'], 'archiver': ['ar', '-cr'], 'ranlib': ['ranlib']}

    def get_flags_linker_so(self):
        if sys.platform == 'darwin':
            return ['-unsharedrts', '-Wl,-bundle,-flat_namespace,-undefined,suppress']
        return BaseNAGFCompiler.get_flags_linker_so(self)

    def get_flags_debug(self):
        version = self.get_version()
        if version and version > '6.1':
            return ['-g', '-u', '-nan', '-C=all', '-thread_safe', '-kind=unique', '-Warn=allocation', '-Warn=subnormal']
        else:
            return ['-g', '-nan', '-C=all', '-u', '-thread_safe']