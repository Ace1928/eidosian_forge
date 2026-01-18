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
def _c_arch_flags(self):
    """ Return detected arch flags from CFLAGS """
    import sysconfig
    try:
        cflags = sysconfig.get_config_vars()['CFLAGS']
    except KeyError:
        return []
    arch_re = re.compile('-arch\\s+(\\w+)')
    arch_flags = []
    for arch in arch_re.findall(cflags):
        arch_flags += ['-arch', arch]
    return arch_flags