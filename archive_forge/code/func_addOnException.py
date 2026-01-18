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
def addOnException(self, handler):
    """Add a handler to be called when an exception occurs in test code.

        This handler cannot affect what result methods are called, and is
        called before any outcome is called on the result object. An example
        use for it is to add some diagnostic state to the test details dict
        which is expensive to calculate and not interesting for reporting in
        the success case.

        Handlers are called before the outcome (such as addFailure) that
        the exception has caused.

        Handlers are called in first-added, first-called order, and if they
        raise an exception, that will propagate out of the test running
        machinery, halting test processing. As a result, do not call code that
        may unreasonably fail.
        """
    self.__exception_handlers.append(handler)