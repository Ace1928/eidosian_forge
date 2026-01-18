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
class TraceMethodCallMeta(type):

    def __init__(self, name, bases, dict):
        for func_name, func in dict.items():
            if inspect.isfunction(func):
                setattr(self, func_name, print_on_call_decorator(func))