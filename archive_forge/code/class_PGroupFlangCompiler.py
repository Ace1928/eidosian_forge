import sys
from numpy.distutils.fcompiler import FCompiler
from sys import platform
from os.path import join, dirname, normpath
import functools
class PGroupFlangCompiler(FCompiler):
    compiler_type = 'flang'
    description = 'Portland Group Fortran LLVM Compiler'
    version_pattern = '\\s*(flang|clang) version (?P<version>[\\d.-]+).*'
    ar_exe = 'lib.exe'
    possible_executables = ['flang']
    executables = {'version_cmd': ['<F77>', '--version'], 'compiler_f77': ['flang'], 'compiler_fix': ['flang'], 'compiler_f90': ['flang'], 'linker_so': [None], 'archiver': [ar_exe, '/verbose', '/OUT:'], 'ranlib': None}
    library_switch = '/OUT:'
    module_dir_switch = '-module '

    def get_libraries(self):
        opt = FCompiler.get_libraries(self)
        opt.extend(['flang', 'flangrti', 'ompstub'])
        return opt

    @functools.lru_cache(maxsize=128)
    def get_library_dirs(self):
        """List of compiler library directories."""
        opt = FCompiler.get_library_dirs(self)
        flang_dir = dirname(self.executables['compiler_f77'][0])
        opt.append(normpath(join(flang_dir, '..', 'lib')))
        return opt

    def get_flags(self):
        return []

    def get_flags_free(self):
        return []

    def get_flags_debug(self):
        return ['-g']

    def get_flags_opt(self):
        return ['-O3']

    def get_flags_arch(self):
        return []

    def runtime_library_dir_option(self, dir):
        raise NotImplementedError