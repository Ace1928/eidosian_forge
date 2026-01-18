import sys
from numpy.distutils.fcompiler import FCompiler
from sys import platform
from os.path import join, dirname, normpath
import functools
class PGroupFCompiler(FCompiler):
    compiler_type = 'pg'
    description = 'Portland Group Fortran Compiler'
    version_pattern = '\\s*pg(f77|f90|hpf|fortran) (?P<version>[\\d.-]+).*'
    if platform == 'darwin':
        executables = {'version_cmd': ['<F77>', '-V'], 'compiler_f77': ['pgfortran', '-dynamiclib'], 'compiler_fix': ['pgfortran', '-Mfixed', '-dynamiclib'], 'compiler_f90': ['pgfortran', '-dynamiclib'], 'linker_so': ['libtool'], 'archiver': ['ar', '-cr'], 'ranlib': ['ranlib']}
        pic_flags = ['']
    else:
        executables = {'version_cmd': ['<F77>', '-V'], 'compiler_f77': ['pgfortran'], 'compiler_fix': ['pgfortran', '-Mfixed'], 'compiler_f90': ['pgfortran'], 'linker_so': ['<F90>'], 'archiver': ['ar', '-cr'], 'ranlib': ['ranlib']}
        pic_flags = ['-fpic']
    module_dir_switch = '-module '
    module_include_switch = '-I'

    def get_flags(self):
        opt = ['-Minform=inform', '-Mnosecond_underscore']
        return self.pic_flags + opt

    def get_flags_opt(self):
        return ['-fast']

    def get_flags_debug(self):
        return ['-g']
    if platform == 'darwin':

        def get_flags_linker_so(self):
            return ['-dynamic', '-undefined', 'dynamic_lookup']
    else:

        def get_flags_linker_so(self):
            return ['-shared', '-fpic']

    def runtime_library_dir_option(self, dir):
        return '-R%s' % dir