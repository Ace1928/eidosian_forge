import gc
import sys
import unittest as pyunit
import weakref
from io import StringIO
from twisted.internet import defer, reactor
from twisted.python.compat import _PYPY
from twisted.python.reflect import namedAny
from twisted.trial import reporter, runner, unittest, util
from twisted.trial._asyncrunner import (
from twisted.trial.test import erroneous
from twisted.trial.test.test_suppression import SuppressionMixin
class SynchronousTodoTests(TodoMixin, unittest.SynchronousTestCase):
    """
    Test the class skipping features in the synchronous case.

    See: L{twisted.trial.test.test_tests.TodoMixin}
    """
    Todo = namedAny('twisted.trial.test.skipping.SynchronousTodo')
    SetUpTodo = namedAny('twisted.trial.test.skipping.SynchronousSetUpTodo')
    TearDownTodo = namedAny('twisted.trial.test.skipping.SynchronousTearDownTodo')