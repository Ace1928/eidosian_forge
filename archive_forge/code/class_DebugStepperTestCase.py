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
class DebugStepperTestCase(DebugTestCase):

    def step(self, varnames_and_values, source_line=None, lineno=None):
        gdb.execute(self.command)
        for varname, value in varnames_and_values:
            self.assertEqual(self.read_var(varname), value, self.local_info())
        self.lineno_equals(source_line, lineno)