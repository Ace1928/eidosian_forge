import os
import re
import sys
import shutil
import warnings
import textwrap
import unittest
import tempfile
import subprocess
from distutils import ccompiler
import runtests
import Cython.Distutils.extension
import Cython.Distutils.old_build_ext as build_ext
from Cython.Debugger import Cygdb as cygdb
class TestAll(GdbDebuggerTestCase):

    def test_all(self):
        if not test_gdb():
            return
        out, err = self.p.communicate()
        out = out.decode('UTF-8')
        err = err.decode('UTF-8')
        exit_status = self.p.returncode
        if exit_status == 1:
            sys.stderr.write(out)
            sys.stderr.write(err)
        elif exit_status >= 2:
            border = u'*' * 30
            start = u'%s   v INSIDE GDB v   %s' % (border, border)
            stderr = u'%s   v STDERR v   %s' % (border, border)
            end = u'%s   ^ INSIDE GDB ^   %s' % (border, border)
            errmsg = u'\n%s\n%s%s\n%s%s' % (start, out, stderr, err, end)
            sys.stderr.write(errmsg)