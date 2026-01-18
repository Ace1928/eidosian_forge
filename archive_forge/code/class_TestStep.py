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
class TestStep(DebugStepperTestCase):
    """
    Test stepping. Stepping happens in the code found in
    Cython/Debugger/Tests/codefile.
    """

    def test_cython_step(self):
        gdb.execute('cy break codefile.spam')
        gdb.execute('run', to_string=True)
        self.lineno_equals('def spam(a=0):')
        gdb.execute('cy step', to_string=True)
        self.lineno_equals('b = c = d = 0')
        self.command = 'cy step'
        self.step([('b', 0)], source_line='b = 1')
        self.step([('b', 1), ('c', 0)], source_line='c = 2')
        self.step([('c', 2)], source_line='int(10)')
        self.step([], source_line='puts("spam")')
        gdb.execute('cont', to_string=True)
        self.assertEqual(len(gdb.inferiors()), 1)
        self.assertEqual(gdb.inferiors()[0].pid, 0)

    def test_c_step(self):
        self.break_and_run('some_c_function()')
        gdb.execute('cy step', to_string=True)
        self.assertEqual(gdb.selected_frame().name(), 'some_c_function')

    def test_python_step(self):
        self.break_and_run('os.path.join("foo", "bar")')
        result = gdb.execute('cy step', to_string=True)
        curframe = gdb.selected_frame()
        self.assertEqual(curframe.name(), 'PyEval_EvalFrameEx')
        pyframe = libpython.Frame(curframe).get_pyop()
        frame_name = pyframe.co_name.proxyval(set())
        self.assertEqual(frame_name, 'join')
        assert re.match('\\d+    def join\\(', result), result