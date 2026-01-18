import sys
import os
import re
import copy
import warnings
import subprocess
import textwrap
from glob import glob
from functools import reduce
from configparser import NoOptionError
from configparser import RawConfigParser as ConfigParser
from distutils.errors import DistutilsError
from distutils.dist import Distribution
import sysconfig
from numpy.distutils import log
from distutils.util import get_platform
from numpy.distutils.exec_command import (
from numpy.distutils.misc_util import (is_sequence, is_string,
from numpy.distutils.command.config import config as cmd_config
from numpy.distutils import customized_ccompiler as _customized_ccompiler
from numpy.distutils import _shell_utils
import distutils.ccompiler
import tempfile
import shutil
import platform
class accelerate_info(system_info):
    section = 'accelerate'
    _lib_names = ['accelerate', 'veclib']
    notfounderror = BlasNotFoundError

    def calc_info(self):
        libraries = os.environ.get('ACCELERATE')
        if libraries:
            libraries = [libraries]
        else:
            libraries = self.get_libs('libraries', self._lib_names)
        libraries = [lib.strip().lower() for lib in libraries]
        if sys.platform == 'darwin' and (not os.getenv('_PYTHON_HOST_PLATFORM', None)):
            args = []
            link_args = []
            if get_platform()[-4:] == 'i386' or 'intel' in get_platform() or 'x86_64' in get_platform() or ('i386' in platform.platform()):
                intel = 1
            else:
                intel = 0
            if os.path.exists('/System/Library/Frameworks/Accelerate.framework/') and 'accelerate' in libraries:
                if intel:
                    args.extend(['-msse3'])
                args.extend(['-I/System/Library/Frameworks/vecLib.framework/Headers'])
                link_args.extend(['-Wl,-framework', '-Wl,Accelerate'])
            elif os.path.exists('/System/Library/Frameworks/vecLib.framework/') and 'veclib' in libraries:
                if intel:
                    args.extend(['-msse3'])
                args.extend(['-I/System/Library/Frameworks/vecLib.framework/Headers'])
                link_args.extend(['-Wl,-framework', '-Wl,vecLib'])
            if args:
                macros = [('NO_ATLAS_INFO', 3), ('HAVE_CBLAS', None), ('ACCELERATE_NEW_LAPACK', None)]
                if os.getenv('NPY_USE_BLAS_ILP64', None):
                    print('Setting HAVE_BLAS_ILP64')
                    macros += [('HAVE_BLAS_ILP64', None), ('ACCELERATE_LAPACK_ILP64', None)]
                self.set_info(extra_compile_args=args, extra_link_args=link_args, define_macros=macros)
        return