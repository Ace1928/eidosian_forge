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
class TestLocalsGlobals(DebugTestCase):

    def test_locals(self):
        self.break_and_run('int(10)')
        result = gdb.execute('cy locals', to_string=True)
        assert 'a = 0', repr(result)
        assert 'b = (int) 1', result
        assert 'c = (int) 2' in result, repr(result)

    def test_globals(self):
        self.break_and_run('int(10)')
        result = gdb.execute('cy globals', to_string=True)
        assert '__name__ ' in result, repr(result)
        assert '__doc__ ' in result, repr(result)
        assert 'os ' in result, repr(result)
        assert 'c_var ' in result, repr(result)
        assert 'python_var ' in result, repr(result)