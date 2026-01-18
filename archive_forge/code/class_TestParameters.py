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
class TestParameters(unittest.TestCase):

    def test_parameters(self):
        gdb.execute('set cy_colorize_code on')
        assert libcython.parameters.colorize_code
        gdb.execute('set cy_colorize_code off')
        assert not libcython.parameters.colorize_code