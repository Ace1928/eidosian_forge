import sys
from numpy.distutils.ccompiler import simple_version_match
from numpy.distutils.fcompiler import FCompiler, dummy_fortran_file
class IntelEM64VisualFCompiler(IntelVisualFCompiler):
    compiler_type = 'intelvem'
    description = 'Intel Visual Fortran Compiler for 64-bit apps'
    version_match = simple_version_match(start='Intel\\(R\\).*?64,')

    def get_flags_arch(self):
        return []