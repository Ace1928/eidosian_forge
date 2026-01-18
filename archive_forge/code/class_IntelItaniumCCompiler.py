import platform
from distutils.unixccompiler import UnixCCompiler
from numpy.distutils.exec_command import find_executable
from numpy.distutils.ccompiler import simple_version_match
class IntelItaniumCCompiler(IntelCCompiler):
    compiler_type = 'intele'
    for cc_exe in map(find_executable, ['icc', 'ecc']):
        if cc_exe:
            break