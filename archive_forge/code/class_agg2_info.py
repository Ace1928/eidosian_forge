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
class agg2_info(system_info):
    section = 'agg2'
    dir_env_var = 'AGG2'

    def get_paths(self, section, key):
        pre_dirs = system_info.get_paths(self, section, key)
        dirs = []
        for d in pre_dirs:
            dirs.extend([d] + self.combine_paths(d, ['agg2*']))
        return [d for d in dirs if os.path.isdir(d)]

    def calc_info(self):
        src_dirs = self.get_src_dirs()
        src_dir = ''
        for d in src_dirs:
            if os.path.isfile(os.path.join(d, 'src', 'agg_affine_matrix.cpp')):
                src_dir = d
                break
        if not src_dir:
            return
        if sys.platform == 'win32':
            agg2_srcs = glob(os.path.join(src_dir, 'src', 'platform', 'win32', 'agg_win32_bmp.cpp'))
        else:
            agg2_srcs = glob(os.path.join(src_dir, 'src', '*.cpp'))
            agg2_srcs += [os.path.join(src_dir, 'src', 'platform', 'X11', 'agg_platform_support.cpp')]
        info = {'libraries': [('agg2_src', {'sources': agg2_srcs, 'include_dirs': [os.path.join(src_dir, 'include')]})], 'include_dirs': [os.path.join(src_dir, 'include')]}
        if info:
            self.set_info(**info)
        return