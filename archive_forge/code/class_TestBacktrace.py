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
class TestBacktrace(DebugTestCase):

    def test_backtrace(self):
        libcython.parameters.colorize_code.value = False
        self.break_and_run('os.path.join("foo", "bar")')

        def match_backtrace_output(result):
            assert re.search('\\#\\d+ *0x.* in spam\\(\\) at .*codefile\\.pyx:22', result), result
            assert 'os.path.join("foo", "bar")' in result, result
        result = gdb.execute('cy bt', to_string=True)
        match_backtrace_output(result)
        result = gdb.execute('cy bt -a', to_string=True)
        match_backtrace_output(result)