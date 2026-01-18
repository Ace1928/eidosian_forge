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
class djbfft_info(system_info):
    section = 'djbfft'
    dir_env_var = 'DJBFFT'
    notfounderror = DJBFFTNotFoundError

    def get_paths(self, section, key):
        pre_dirs = system_info.get_paths(self, section, key)
        dirs = []
        for d in pre_dirs:
            dirs.extend(self.combine_paths(d, ['djbfft']) + [d])
        return [d for d in dirs if os.path.isdir(d)]

    def calc_info(self):
        lib_dirs = self.get_lib_dirs()
        incl_dirs = self.get_include_dirs()
        info = None
        for d in lib_dirs:
            p = self.combine_paths(d, ['djbfft.a'])
            if p:
                info = {'extra_objects': p}
                break
            p = self.combine_paths(d, ['libdjbfft.a', 'libdjbfft' + so_ext])
            if p:
                info = {'libraries': ['djbfft'], 'library_dirs': [d]}
                break
        if info is None:
            return
        for d in incl_dirs:
            if len(self.combine_paths(d, ['fftc8.h', 'fftfreq.h'])) == 2:
                dict_append(info, include_dirs=[d], define_macros=[('SCIPY_DJBFFT_H', None)])
                self.set_info(**info)
                return
        return