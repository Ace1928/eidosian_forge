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
class f2py_info(system_info):

    def calc_info(self):
        try:
            import numpy.f2py as f2py
        except ImportError:
            return
        f2py_dir = os.path.join(os.path.dirname(f2py.__file__), 'src')
        self.set_info(sources=[os.path.join(f2py_dir, 'fortranobject.c')], include_dirs=[f2py_dir])
        return