import os
import shutil
import pytest
from tempfile import mkstemp, mkdtemp
from subprocess import Popen, PIPE
import importlib.metadata
from distutils.errors import DistutilsError
from numpy.testing import assert_, assert_equal, assert_raises
from numpy.distutils import ccompiler, customized_ccompiler
from numpy.distutils.system_info import system_info, ConfigParser, mkl_info
from numpy.distutils.system_info import AliasedOptionError
from numpy.distutils.system_info import default_lib_dirs, default_include_dirs
from numpy.distutils import _shell_utils
class _system_info(system_info):

    def __init__(self, default_lib_dirs=default_lib_dirs, default_include_dirs=default_include_dirs, verbosity=1):
        self.__class__.info = {}
        self.local_prefixes = []
        defaults = {'library_dirs': '', 'include_dirs': '', 'runtime_library_dirs': '', 'rpath': '', 'src_dirs': '', 'search_static_first': '0', 'extra_compile_args': '', 'extra_link_args': ''}
        self.cp = ConfigParser(defaults)

    def _check_libs(self, lib_dirs, libs, opt_libs, exts):
        """Override _check_libs to return with all dirs """
        info = {'libraries': libs, 'library_dirs': lib_dirs}
        return info