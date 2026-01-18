import os
import re
import sys
import platform
import shlex
import time
import subprocess
from copy import copy
from pathlib import Path
from distutils import ccompiler
from distutils.ccompiler import (
from distutils.errors import (
from distutils.sysconfig import customize_compiler
from distutils.version import LooseVersion
from numpy.distutils import log
from numpy.distutils.exec_command import (
from numpy.distutils.misc_util import cyg2win32, is_sequence, mingw32, \
import threading
def CCompiler_customize_cmd(self, cmd, ignore=()):
    """
    Customize compiler using distutils command.

    Parameters
    ----------
    cmd : class instance
        An instance inheriting from `distutils.cmd.Command`.
    ignore : sequence of str, optional
        List of `CCompiler` commands (without ``'set_'``) that should not be
        altered. Strings that are checked for are:
        ``('include_dirs', 'define', 'undef', 'libraries', 'library_dirs',
        'rpath', 'link_objects')``.

    Returns
    -------
    None

    """
    log.info('customize %s using %s' % (self.__class__.__name__, cmd.__class__.__name__))
    if hasattr(self, 'compiler') and 'clang' in self.compiler[0] and (not (platform.machine() == 'arm64' and sys.platform == 'darwin')):
        self.compiler.append('-ftrapping-math')
        self.compiler_so.append('-ftrapping-math')

    def allow(attr):
        return getattr(cmd, attr, None) is not None and attr not in ignore
    if allow('include_dirs'):
        self.set_include_dirs(cmd.include_dirs)
    if allow('define'):
        for name, value in cmd.define:
            self.define_macro(name, value)
    if allow('undef'):
        for macro in cmd.undef:
            self.undefine_macro(macro)
    if allow('libraries'):
        self.set_libraries(self.libraries + cmd.libraries)
    if allow('library_dirs'):
        self.set_library_dirs(self.library_dirs + cmd.library_dirs)
    if allow('rpath'):
        self.set_runtime_library_dirs(cmd.rpath)
    if allow('link_objects'):
        self.set_link_objects(cmd.link_objects)