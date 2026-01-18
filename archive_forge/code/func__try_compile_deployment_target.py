import sys
import os
from io import StringIO
import textwrap
from distutils.core import Distribution
from distutils.command.build_ext import build_ext
from distutils import sysconfig
from distutils.tests.support import (TempdirManager, LoggingSilencer,
from distutils.extension import Extension
from distutils.errors import (
import unittest
from test import support
from test.support import os_helper
from test.support.script_helper import assert_python_ok
from test.support import threading_helper
def _try_compile_deployment_target(self, operator, target):
    orig_environ = os.environ
    os.environ = orig_environ.copy()
    self.addCleanup(setattr, os, 'environ', orig_environ)
    if target is None:
        if os.environ.get('MACOSX_DEPLOYMENT_TARGET'):
            del os.environ['MACOSX_DEPLOYMENT_TARGET']
    else:
        os.environ['MACOSX_DEPLOYMENT_TARGET'] = target
    deptarget_c = os.path.join(self.tmp_dir, 'deptargetmodule.c')
    with open(deptarget_c, 'w') as fp:
        fp.write(textwrap.dedent('                #include <AvailabilityMacros.h>\n\n                int dummy;\n\n                #if TARGET %s MAC_OS_X_VERSION_MIN_REQUIRED\n                #else\n                #error "Unexpected target"\n                #endif\n\n            ' % operator))
    target = sysconfig.get_config_var('MACOSX_DEPLOYMENT_TARGET')
    target = tuple(map(int, target.split('.')[0:2]))
    if target[:2] < (10, 10):
        target = '%02d%01d0' % target
    elif len(target) >= 2:
        target = '%02d%02d00' % target
    else:
        target = '%02d0000' % target
    deptarget_ext = Extension('deptarget', [deptarget_c], extra_compile_args=['-DTARGET=%s' % (target,)])
    dist = Distribution({'name': 'deptarget', 'ext_modules': [deptarget_ext]})
    dist.package_dir = self.tmp_dir
    cmd = self.build_ext(dist)
    cmd.build_lib = self.tmp_dir
    cmd.build_temp = self.tmp_dir
    try:
        old_stdout = sys.stdout
        if not support.verbose:
            sys.stdout = StringIO()
        try:
            cmd.ensure_finalized()
            cmd.run()
        finally:
            sys.stdout = old_stdout
    except CompileError:
        self.fail('Wrong deployment target during compilation')