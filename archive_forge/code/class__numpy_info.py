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
class _numpy_info(system_info):
    section = 'Numeric'
    modulename = 'Numeric'
    notfounderror = NumericNotFoundError

    def __init__(self):
        include_dirs = []
        try:
            module = __import__(self.modulename)
            prefix = []
            for name in module.__file__.split(os.sep):
                if name == 'lib':
                    break
                prefix.append(name)
            try:
                include_dirs.append(getattr(module, 'get_include')())
            except AttributeError:
                pass
            include_dirs.append(sysconfig.get_path('include'))
        except ImportError:
            pass
        py_incl_dir = sysconfig.get_path('include')
        include_dirs.append(py_incl_dir)
        py_pincl_dir = sysconfig.get_path('platinclude')
        if py_pincl_dir not in include_dirs:
            include_dirs.append(py_pincl_dir)
        for d in default_include_dirs:
            d = os.path.join(d, os.path.basename(py_incl_dir))
            if d not in include_dirs:
                include_dirs.append(d)
        system_info.__init__(self, default_lib_dirs=[], default_include_dirs=include_dirs)

    def calc_info(self):
        try:
            module = __import__(self.modulename)
        except ImportError:
            return
        info = {}
        macros = []
        for v in ['__version__', 'version']:
            vrs = getattr(module, v, None)
            if vrs is None:
                continue
            macros = [(self.modulename.upper() + '_VERSION', _c_string_literal(vrs)), (self.modulename.upper(), None)]
            break
        dict_append(info, define_macros=macros)
        include_dirs = self.get_include_dirs()
        inc_dir = None
        for d in include_dirs:
            if self.combine_paths(d, os.path.join(self.modulename, 'arrayobject.h')):
                inc_dir = d
                break
        if inc_dir is not None:
            dict_append(info, include_dirs=[inc_dir])
        if info:
            self.set_info(**info)
        return