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
def _testDeprecatedFailMethods(self):
    """Test that the deprecated fail* methods get removed in 3.x"""
    if sys.version_info[:2] < (3, 3):
        return
    deprecated_names = ['failIfEqual', 'failUnlessEqual', 'failUnlessAlmostEqual', 'failIfAlmostEqual', 'failUnless', 'failUnlessRaises', 'failIf', 'assertDictContainsSubset']
    for deprecated_name in deprecated_names:
        with self.assertRaises(AttributeError):
            getattr(self, deprecated_name)