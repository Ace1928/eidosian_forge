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
class amd_info(system_info):
    section = 'amd'
    dir_env_var = 'AMD'
    _lib_names = ['amd']

    def calc_info(self):
        lib_dirs = self.get_lib_dirs()
        opt = self.get_option_single('amd_libs', 'libraries')
        amd_libs = self.get_libs(opt, self._lib_names)
        info = self.check_libs(lib_dirs, amd_libs, [])
        if info is None:
            return
        include_dirs = self.get_include_dirs()
        inc_dir = None
        for d in include_dirs:
            p = self.combine_paths(d, 'amd.h')
            if p:
                inc_dir = os.path.dirname(p[0])
                break
        if inc_dir is not None:
            dict_append(info, include_dirs=[inc_dir], define_macros=[('SCIPY_AMD_H', None)], swig_opts=['-I' + inc_dir])
        self.set_info(**info)
        return