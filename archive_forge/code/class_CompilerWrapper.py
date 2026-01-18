import sys
import unittest
from test.support.os_helper import EnvironmentVarGuard
from distutils import sysconfig
from distutils.unixccompiler import UnixCCompiler
class CompilerWrapper(UnixCCompiler):

    def rpath_foo(self):
        return self.runtime_library_dir_option('/foo')