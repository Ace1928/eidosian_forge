import os
import re
import sys
import platform
import shlex
import time
import subprocess
from copy import copy
from pathlib import Path
from distutils import ccompiler
from distutils.ccompiler import (
from distutils.errors import (
from distutils.sysconfig import customize_compiler
from distutils.version import LooseVersion
from numpy.distutils import log
from numpy.distutils.exec_command import (
from numpy.distutils.misc_util import cyg2win32, is_sequence, mingw32, \
import threading
def CCompiler_cxx_compiler(self):
    """
    Return the C++ compiler.

    Parameters
    ----------
    None

    Returns
    -------
    cxx : class instance
        The C++ compiler, as a `CCompiler` instance.

    """
    if self.compiler_type in ('msvc', 'intelw', 'intelemw'):
        return self
    cxx = copy(self)
    cxx.compiler_cxx = cxx.compiler_cxx
    cxx.compiler_so = [cxx.compiler_cxx[0]] + sanitize_cxx_flags(cxx.compiler_so[1:])
    if sys.platform.startswith(('aix', 'os400')) and 'ld_so_aix' in cxx.linker_so[0]:
        cxx.linker_so = [cxx.linker_so[0], cxx.compiler_cxx[0]] + cxx.linker_so[2:]
    if sys.platform.startswith('os400'):
        cxx.compiler_so.append('-D__STDC_FORMAT_MACROS')
        cxx.compiler_so.append('-fno-extern-tls-init')
        cxx.linker_so.append('-fno-extern-tls-init')
    else:
        cxx.linker_so = [cxx.compiler_cxx[0]] + cxx.linker_so[1:]
    return cxx