from numpy.distutils.fcompiler import FCompiler
class PathScaleFCompiler(FCompiler):
    compiler_type = 'pathf95'
    description = 'PathScale Fortran Compiler'
    version_pattern = 'PathScale\\(TM\\) Compiler Suite: Version (?P<version>[\\d.]+)'
    executables = {'version_cmd': ['pathf95', '-version'], 'compiler_f77': ['pathf95', '-fixedform'], 'compiler_fix': ['pathf95', '-fixedform'], 'compiler_f90': ['pathf95'], 'linker_so': ['pathf95', '-shared'], 'archiver': ['ar', '-cr'], 'ranlib': ['ranlib']}
    pic_flags = ['-fPIC']
    module_dir_switch = '-module '
    module_include_switch = '-I'

    def get_flags_opt(self):
        return ['-O3']

    def get_flags_debug(self):
        return ['-g']