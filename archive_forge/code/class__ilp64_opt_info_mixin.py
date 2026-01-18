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
class _ilp64_opt_info_mixin:
    symbol_suffix = None
    symbol_prefix = None

    def _check_info(self, info):
        macros = dict(info.get('define_macros', []))
        prefix = macros.get('BLAS_SYMBOL_PREFIX', '')
        suffix = macros.get('BLAS_SYMBOL_SUFFIX', '')
        if self.symbol_prefix not in (None, prefix):
            return False
        if self.symbol_suffix not in (None, suffix):
            return False
        return bool(info)