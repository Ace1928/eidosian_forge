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
class _pkg_config_info(system_info):
    section = None
    config_env_var = 'PKG_CONFIG'
    default_config_exe = 'pkg-config'
    append_config_exe = ''
    version_macro_name = None
    release_macro_name = None
    version_flag = '--modversion'
    cflags_flag = '--cflags'

    def get_config_exe(self):
        if self.config_env_var in os.environ:
            return os.environ[self.config_env_var]
        return self.default_config_exe

    def get_config_output(self, config_exe, option):
        cmd = config_exe + ' ' + self.append_config_exe + ' ' + option
        try:
            o = subprocess.check_output(cmd)
        except (OSError, subprocess.CalledProcessError):
            pass
        else:
            o = filepath_from_subprocess_output(o)
            return o

    def calc_info(self):
        config_exe = find_executable(self.get_config_exe())
        if not config_exe:
            log.warn('File not found: %s. Cannot determine %s info.' % (config_exe, self.section))
            return
        info = {}
        macros = []
        libraries = []
        library_dirs = []
        include_dirs = []
        extra_link_args = []
        extra_compile_args = []
        version = self.get_config_output(config_exe, self.version_flag)
        if version:
            macros.append((self.__class__.__name__.split('.')[-1].upper(), _c_string_literal(version)))
            if self.version_macro_name:
                macros.append((self.version_macro_name + '_%s' % version.replace('.', '_'), None))
        if self.release_macro_name:
            release = self.get_config_output(config_exe, '--release')
            if release:
                macros.append((self.release_macro_name + '_%s' % release.replace('.', '_'), None))
        opts = self.get_config_output(config_exe, '--libs')
        if opts:
            for opt in opts.split():
                if opt[:2] == '-l':
                    libraries.append(opt[2:])
                elif opt[:2] == '-L':
                    library_dirs.append(opt[2:])
                else:
                    extra_link_args.append(opt)
        opts = self.get_config_output(config_exe, self.cflags_flag)
        if opts:
            for opt in opts.split():
                if opt[:2] == '-I':
                    include_dirs.append(opt[2:])
                elif opt[:2] == '-D':
                    if '=' in opt:
                        n, v = opt[2:].split('=')
                        macros.append((n, v))
                    else:
                        macros.append((opt[2:], None))
                else:
                    extra_compile_args.append(opt)
        if macros:
            dict_append(info, define_macros=macros)
        if libraries:
            dict_append(info, libraries=libraries)
        if library_dirs:
            dict_append(info, library_dirs=library_dirs)
        if include_dirs:
            dict_append(info, include_dirs=include_dirs)
        if extra_link_args:
            dict_append(info, extra_link_args=extra_link_args)
        if extra_compile_args:
            dict_append(info, extra_compile_args=extra_compile_args)
        if info:
            self.set_info(**info)
        return