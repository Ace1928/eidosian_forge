from __future__ import absolute_import
import os
import re
import sys
import trace
import inspect
import warnings
import unittest
import textwrap
import tempfile
import functools
import traceback
import itertools
import gdb
from .. import libcython
from .. import libpython
from . import TestLibCython as test_libcython
from ...Utils import add_metaclass
class TestExec(DebugTestCase):

    def setUp(self):
        super(TestExec, self).setUp()
        self.fd, self.tmpfilename = tempfile.mkstemp()
        self.tmpfile = os.fdopen(self.fd, 'r+')

    def tearDown(self):
        super(TestExec, self).tearDown()
        try:
            self.tmpfile.close()
        finally:
            os.remove(self.tmpfilename)

    def eval_command(self, command):
        gdb.execute('cy exec open(%r, "w").write(str(%s))' % (self.tmpfilename, command))
        return self.tmpfile.read().strip()

    def test_cython_exec(self):
        self.break_and_run('os.path.join("foo", "bar")')
        self.assertEqual('[0]', self.eval_command('[a]'))
        return
        result = gdb.execute(textwrap.dedent('            cy exec\n            pass\n\n            "nothing"\n            end\n            '))
        result = self.tmpfile.read().rstrip()
        self.assertEqual('', result)

    def test_python_exec(self):
        self.break_and_run('os.path.join("foo", "bar")')
        gdb.execute('cy step')
        gdb.execute('cy exec some_random_var = 14')
        self.assertEqual('14', self.eval_command('some_random_var'))