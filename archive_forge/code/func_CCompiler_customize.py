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
def CCompiler_customize(self, dist, need_cxx=0):
    """
    Do any platform-specific customization of a compiler instance.

    This method calls `distutils.sysconfig.customize_compiler` for
    platform-specific customization, as well as optionally remove a flag
    to suppress spurious warnings in case C++ code is being compiled.

    Parameters
    ----------
    dist : object
        This parameter is not used for anything.
    need_cxx : bool, optional
        Whether or not C++ has to be compiled. If so (True), the
        ``"-Wstrict-prototypes"`` option is removed to prevent spurious
        warnings. Default is False.

    Returns
    -------
    None

    Notes
    -----
    All the default options used by distutils can be extracted with::

      from distutils import sysconfig
      sysconfig.get_config_vars('CC', 'CXX', 'OPT', 'BASECFLAGS',
                                'CCSHARED', 'LDSHARED', 'SO')

    """
    log.info('customize %s' % self.__class__.__name__)
    customize_compiler(self)
    if need_cxx:
        try:
            self.compiler_so.remove('-Wstrict-prototypes')
        except (AttributeError, ValueError):
            pass
        if hasattr(self, 'compiler') and 'cc' in self.compiler[0]:
            if not self.compiler_cxx:
                if self.compiler[0].startswith('gcc'):
                    a, b = ('gcc', 'g++')
                else:
                    a, b = ('cc', 'c++')
                self.compiler_cxx = [self.compiler[0].replace(a, b)] + self.compiler[1:]
        else:
            if hasattr(self, 'compiler'):
                log.warn('#### %s #######' % (self.compiler,))
            if not hasattr(self, 'compiler_cxx'):
                log.warn('Missing compiler_cxx fix for ' + self.__class__.__name__)
    if hasattr(self, 'compiler') and ('gcc' in self.compiler[0] or 'g++' in self.compiler[0] or 'clang' in self.compiler[0]):
        self._auto_depends = True
    elif os.name == 'posix':
        import tempfile
        import shutil
        tmpdir = tempfile.mkdtemp()
        try:
            fn = os.path.join(tmpdir, 'file.c')
            with open(fn, 'w') as f:
                f.write('int a;\n')
            self.compile([fn], output_dir=tmpdir, extra_preargs=['-MMD', '-MF', fn + '.d'])
            self._auto_depends = True
        except CompileError:
            self._auto_depends = False
        finally:
            shutil.rmtree(tmpdir)
    return