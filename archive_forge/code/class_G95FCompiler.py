from numpy.distutils.fcompiler import FCompiler
class G95FCompiler(FCompiler):
    compiler_type = 'g95'
    description = 'G95 Fortran Compiler'
    version_pattern = 'G95 \\((GCC (?P<gccversion>[\\d.]+)|.*?) \\(g95 (?P<version>.*)!\\) (?P<date>.*)\\).*'
    executables = {'version_cmd': ['<F90>', '--version'], 'compiler_f77': ['g95', '-ffixed-form'], 'compiler_fix': ['g95', '-ffixed-form'], 'compiler_f90': ['g95'], 'linker_so': ['<F90>', '-shared'], 'archiver': ['ar', '-cr'], 'ranlib': ['ranlib']}
    pic_flags = ['-fpic']
    module_dir_switch = '-fmod='
    module_include_switch = '-I'

    def get_flags(self):
        return ['-fno-second-underscore']

    def get_flags_opt(self):
        return ['-O']

    def get_flags_debug(self):
        return ['-g']