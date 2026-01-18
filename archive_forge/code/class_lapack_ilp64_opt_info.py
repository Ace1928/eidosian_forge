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
class lapack_ilp64_opt_info(lapack_opt_info, _ilp64_opt_info_mixin):
    notfounderror = LapackILP64NotFoundError
    lapack_order = ['openblas64_', 'openblas_ilp64', 'accelerate']
    order_env_var_name = 'NPY_LAPACK_ILP64_ORDER'

    def _calc_info(self, name):
        print('lapack_ilp64_opt_info._calc_info(name=%s)' % name)
        info = get_info(name + '_lapack')
        if self._check_info(info):
            self.set_info(**info)
            return True
        else:
            print('%s_lapack does not exist' % name)
        return False