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
class blis_info(blas_info):
    section = 'blis'
    dir_env_var = 'BLIS'
    _lib_names = ['blis']
    notfounderror = BlasNotFoundError

    def calc_info(self):
        lib_dirs = self.get_lib_dirs()
        opt = self.get_option_single('blis_libs', 'libraries')
        blis_libs = self.get_libs(opt, self._lib_names)
        info = self.check_libs2(lib_dirs, blis_libs, [])
        if info is None:
            return
        incl_dirs = self.get_include_dirs()
        dict_append(info, language='c', define_macros=[('HAVE_CBLAS', None)], include_dirs=incl_dirs)
        self.set_info(**info)