import sys
from numpy.distutils.ccompiler import simple_version_match
from numpy.distutils.fcompiler import FCompiler, dummy_fortran_file
class IntelItaniumFCompiler(IntelFCompiler):
    compiler_type = 'intele'
    compiler_aliases = ()
    description = 'Intel Fortran Compiler for Itanium apps'
    version_match = intel_version_match('Itanium|IA-64')
    possible_executables = ['ifort', 'efort', 'efc']
    executables = {'version_cmd': None, 'compiler_f77': [None, '-FI', '-w90', '-w95'], 'compiler_fix': [None, '-FI'], 'compiler_f90': [None], 'linker_so': ['<F90>', '-shared'], 'archiver': ['ar', '-cr'], 'ranlib': ['ranlib']}