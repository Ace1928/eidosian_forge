import copy
import functools
import itertools
import sys
import types
import unittest
import warnings
from testtools.compat import reraise
from testtools import content
from testtools.helpers import try_import
from testtools.matchers import (
from testtools.matchers._basic import _FlippedEquals
from testtools.monkey import patch
from testtools.runtest import (
from testtools.testresult import (
class WithAttributes:
    """A mix-in class for modifying test id by attributes.

    e.g.
    >>> class MyTest(WithAttributes, TestCase):
    ...    @attr('foo')
    ...    def test_bar(self):
    ...        pass
    >>> MyTest('test_bar').id()
    testtools.testcase.MyTest/test_bar[foo]
    """

    def id(self):
        orig = super().id()
        fn = self._get_test_method()
        attributes = getattr(fn, '__testtools_attrs', None)
        if not attributes:
            return orig
        return orig + '[' + ','.join(sorted(attributes)) + ']'