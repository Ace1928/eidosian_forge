import contextlib
import difflib
import pprint
import pickle
import re
import sys
import logging
import warnings
import weakref
import inspect
import types
from copy import deepcopy
from test import support
import unittest
from unittest.test.support import (
from test.support import captured_stderr, gc_collect
class LoggingTestCase(unittest.TestCase):
    """A test case which logs its calls."""

    def __init__(self, events):
        super(Test.LoggingTestCase, self).__init__('test')
        self.events = events

    def setUp(self):
        self.events.append('setUp')

    def test(self):
        self.events.append('test')

    def tearDown(self):
        self.events.append('tearDown')