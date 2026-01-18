from numpy.distutils.fcompiler import FCompiler
from numpy.distutils import customized_fcompiler
class NoneFCompiler(FCompiler):
    compiler_type = 'none'
    description = 'Fake Fortran compiler'
    executables = {'compiler_f77': None, 'compiler_f90': None, 'compiler_fix': None, 'linker_so': None, 'linker_exe': None, 'archiver': None, 'ranlib': None, 'version_cmd': None}

    def find_executables(self):
        pass