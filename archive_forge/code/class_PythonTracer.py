from __future__ import print_function
import sys
import greenlet
import unittest
from . import TestCase
from . import PY312
class PythonTracer(object):
    oldtrace = None

    def __init__(self):
        self.actions = []

    def __call__(self, frame, event, arg):
        self.actions.append((event, frame.f_code.co_name))

    def __enter__(self):
        self.oldtrace = sys.setprofile(self)
        return self.actions

    def __exit__(self, *args):
        sys.setprofile(self.oldtrace)