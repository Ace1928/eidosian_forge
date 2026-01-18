import sys
from numpy.distutils.ccompiler import simple_version_match
from numpy.distutils.fcompiler import FCompiler, dummy_fortran_file
class BaseIntelFCompiler(FCompiler):

    def update_executables(self):
        f = dummy_fortran_file()
        self.executables['version_cmd'] = ['<F77>', '-FI', '-V', '-c', f + '.f', '-o', f + '.o']

    def runtime_library_dir_option(self, dir):
        assert ',' not in dir
        return '-Wl,-rpath=%s' % dir