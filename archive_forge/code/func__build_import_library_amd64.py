import os
import sys
import subprocess
import re
import textwrap
import numpy.distutils.ccompiler  # noqa: F401
from numpy.distutils import log
import distutils.cygwinccompiler
from distutils.unixccompiler import UnixCCompiler
from distutils.msvccompiler import get_build_version as get_build_msvc_version
from distutils.errors import UnknownFileError
from numpy.distutils.misc_util import (msvc_runtime_library,
def _build_import_library_amd64():
    out_exists, out_file = _check_for_import_lib()
    if out_exists:
        log.debug('Skip building import library: "%s" exists', out_file)
        return
    dll_file = find_python_dll()
    log.info('Building import library (arch=AMD64): "%s" (from %s)' % (out_file, dll_file))
    def_name = 'python%d%d.def' % tuple(sys.version_info[:2])
    def_file = os.path.join(sys.prefix, 'libs', def_name)
    generate_def(dll_file, def_file)
    cmd = ['dlltool', '-d', def_file, '-l', out_file]
    subprocess.check_call(cmd)