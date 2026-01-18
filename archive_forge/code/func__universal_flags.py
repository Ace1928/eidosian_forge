import re
import os
import sys
import warnings
import platform
import tempfile
import hashlib
import base64
import subprocess
from subprocess import Popen, PIPE, STDOUT
from numpy.distutils.exec_command import filepath_from_subprocess_output
from numpy.distutils.fcompiler import FCompiler
from distutils.version import LooseVersion
def _universal_flags(self, cmd):
    """Return a list of -arch flags for every supported architecture."""
    if not sys.platform == 'darwin':
        return []
    arch_flags = []
    c_archs = self._c_arch_flags()
    if 'i386' in c_archs:
        c_archs[c_archs.index('i386')] = 'i686'
    for arch in ['ppc', 'i686', 'x86_64', 'ppc64', 's390x']:
        if _can_target(cmd, arch) and arch in c_archs:
            arch_flags.extend(['-arch', arch])
    return arch_flags