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
def _find_dll_in_path(dll_name):
    for path in [sys.prefix] + os.environ['PATH'].split(';'):
        filepath = os.path.join(path, dll_name)
        if os.path.exists(filepath):
            return os.path.abspath(filepath)