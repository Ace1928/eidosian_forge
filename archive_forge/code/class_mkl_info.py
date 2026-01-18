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
class mkl_info(system_info):
    section = 'mkl'
    dir_env_var = 'MKLROOT'
    _lib_mkl = ['mkl_rt']

    def get_mkl_rootdir(self):
        mklroot = os.environ.get('MKLROOT', None)
        if mklroot is not None:
            return mklroot
        paths = os.environ.get('LD_LIBRARY_PATH', '').split(os.pathsep)
        ld_so_conf = '/etc/ld.so.conf'
        if os.path.isfile(ld_so_conf):
            with open(ld_so_conf) as f:
                for d in f:
                    d = d.strip()
                    if d:
                        paths.append(d)
        intel_mkl_dirs = []
        for path in paths:
            path_atoms = path.split(os.sep)
            for m in path_atoms:
                if m.startswith('mkl'):
                    d = os.sep.join(path_atoms[:path_atoms.index(m) + 2])
                    intel_mkl_dirs.append(d)
                    break
        for d in paths:
            dirs = glob(os.path.join(d, 'mkl', '*'))
            dirs += glob(os.path.join(d, 'mkl*'))
            for sub_dir in dirs:
                if os.path.isdir(os.path.join(sub_dir, 'lib')):
                    return sub_dir
        return None

    def __init__(self):
        mklroot = self.get_mkl_rootdir()
        if mklroot is None:
            system_info.__init__(self)
        else:
            from .cpuinfo import cpu
            if cpu.is_Itanium():
                plt = '64'
            elif cpu.is_Intel() and cpu.is_64bit():
                plt = 'intel64'
            else:
                plt = '32'
            system_info.__init__(self, default_lib_dirs=[os.path.join(mklroot, 'lib', plt)], default_include_dirs=[os.path.join(mklroot, 'include')])

    def calc_info(self):
        lib_dirs = self.get_lib_dirs()
        incl_dirs = self.get_include_dirs()
        opt = self.get_option_single('mkl_libs', 'libraries')
        mkl_libs = self.get_libs(opt, self._lib_mkl)
        info = self.check_libs2(lib_dirs, mkl_libs)
        if info is None:
            return
        dict_append(info, define_macros=[('SCIPY_MKL_H', None), ('HAVE_CBLAS', None)], include_dirs=incl_dirs)
        if sys.platform == 'win32':
            pass
        else:
            dict_append(info, libraries=['pthread'])
        self.set_info(**info)