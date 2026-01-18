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
def _find_dll_in_winsxs(dll_name):
    winsxs_path = os.path.join(os.environ.get('WINDIR', 'C:\\WINDOWS'), 'winsxs')
    if not os.path.exists(winsxs_path):
        return None
    for root, dirs, files in os.walk(winsxs_path):
        if dll_name in files and arch in root:
            return os.path.join(root, dll_name)
    return None