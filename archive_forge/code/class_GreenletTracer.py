from __future__ import print_function
import sys
import greenlet
import unittest
from . import TestCase
from . import PY312
class GreenletTracer(object):
    oldtrace = None

    def __init__(self, error_on_trace=False):
        self.actions = []
        self.error_on_trace = error_on_trace

    def __call__(self, *args):
        self.actions.append(args)
        if self.error_on_trace:
            raise SomeError

    def __enter__(self):
        self.oldtrace = greenlet.settrace(self)
        return self.actions

    def __exit__(self, *args):
        greenlet.settrace(self.oldtrace)