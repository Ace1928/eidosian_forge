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
class umfpack_info(system_info):
    section = 'umfpack'
    dir_env_var = 'UMFPACK'
    notfounderror = UmfpackNotFoundError
    _lib_names = ['umfpack']

    def calc_info(self):
        lib_dirs = self.get_lib_dirs()
        opt = self.get_option_single('umfpack_libs', 'libraries')
        umfpack_libs = self.get_libs(opt, self._lib_names)
        info = self.check_libs(lib_dirs, umfpack_libs, [])
        if info is None:
            return
        include_dirs = self.get_include_dirs()
        inc_dir = None
        for d in include_dirs:
            p = self.combine_paths(d, ['', 'umfpack'], 'umfpack.h')
            if p:
                inc_dir = os.path.dirname(p[0])
                break
        if inc_dir is not None:
            dict_append(info, include_dirs=[inc_dir], define_macros=[('SCIPY_UMFPACK_H', None)], swig_opts=['-I' + inc_dir])
        dict_append(info, **get_info('amd'))
        self.set_info(**info)
        return