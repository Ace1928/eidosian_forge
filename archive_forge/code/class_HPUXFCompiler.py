from numpy.distutils.fcompiler import FCompiler
class HPUXFCompiler(FCompiler):
    compiler_type = 'hpux'
    description = 'HP Fortran 90 Compiler'
    version_pattern = 'HP F90 (?P<version>[^\\s*,]*)'
    executables = {'version_cmd': ['f90', '+version'], 'compiler_f77': ['f90'], 'compiler_fix': ['f90'], 'compiler_f90': ['f90'], 'linker_so': ['ld', '-b'], 'archiver': ['ar', '-cr'], 'ranlib': ['ranlib']}
    module_dir_switch = None
    module_include_switch = None
    pic_flags = ['+Z']

    def get_flags(self):
        return self.pic_flags + ['+ppu', '+DD64']

    def get_flags_opt(self):
        return ['-O3']

    def get_libraries(self):
        return ['m']

    def get_library_dirs(self):
        opt = ['/usr/lib/hpux64']
        return opt

    def get_version(self, force=0, ok_status=[256, 0, 1]):
        return FCompiler.get_version(self, force, ok_status)