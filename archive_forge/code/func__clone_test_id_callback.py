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
def _clone_test_id_callback(test, callback):
    """Copy a `TestCase`, and make it call callback for its id().

    This is only expected to be used on tests that have been constructed but
    not executed.

    :param test: A TestCase instance.
    :param callback: A callable that takes no parameters and returns a string.
    :return: A copy.copy of the test with id=callback.
    """
    newTest = copy.copy(test)
    newTest.id = callback
    return newTest