import sys
from numpy.distutils.ccompiler import simple_version_match
from numpy.distutils.fcompiler import FCompiler, dummy_fortran_file
class IntelVisualFCompiler(BaseIntelFCompiler):
    compiler_type = 'intelv'
    description = 'Intel Visual Fortran Compiler for 32-bit apps'
    version_match = intel_version_match('32-bit|IA-32')

    def update_executables(self):
        f = dummy_fortran_file()
        self.executables['version_cmd'] = ['<F77>', '/FI', '/c', f + '.f', '/o', f + '.o']
    ar_exe = 'lib.exe'
    possible_executables = ['ifort', 'ifl']
    executables = {'version_cmd': None, 'compiler_f77': [None], 'compiler_fix': [None], 'compiler_f90': [None], 'linker_so': [None], 'archiver': [ar_exe, '/verbose', '/OUT:'], 'ranlib': None}
    compile_switch = '/c '
    object_switch = '/Fo'
    library_switch = '/OUT:'
    module_dir_switch = '/module:'
    module_include_switch = '/I'

    def get_flags(self):
        opt = ['/nologo', '/MD', '/nbs', '/names:lowercase', '/assume:underscore', '/fpp']
        return opt

    def get_flags_free(self):
        return []

    def get_flags_debug(self):
        return ['/4Yb', '/d2']

    def get_flags_opt(self):
        return ['/O1', '/assume:minus0']

    def get_flags_arch(self):
        return ['/arch:IA32', '/QaxSSE3']

    def runtime_library_dir_option(self, dir):
        raise NotImplementedError