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
class x11_info(system_info):
    section = 'x11'
    notfounderror = X11NotFoundError
    _lib_names = ['X11']

    def __init__(self):
        system_info.__init__(self, default_lib_dirs=default_x11_lib_dirs, default_include_dirs=default_x11_include_dirs)

    def calc_info(self):
        if sys.platform in ['win32']:
            return
        lib_dirs = self.get_lib_dirs()
        include_dirs = self.get_include_dirs()
        opt = self.get_option_single('x11_libs', 'libraries')
        x11_libs = self.get_libs(opt, self._lib_names)
        info = self.check_libs(lib_dirs, x11_libs, [])
        if info is None:
            return
        inc_dir = None
        for d in include_dirs:
            if self.combine_paths(d, 'X11/X.h'):
                inc_dir = d
                break
        if inc_dir is not None:
            dict_append(info, include_dirs=[inc_dir])
        self.set_info(**info)