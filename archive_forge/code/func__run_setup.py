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
def _run_setup(self, result):
    """Run the setUp function for this test.

        :param result: A testtools.TestResult to report activity to.
        :raises ValueError: If the base class setUp is not called, a
            ValueError is raised.
        """
    ret = self.setUp()
    if not self.__setup_called:
        raise ValueError('In File: %s\nTestCase.setUp was not called. Have you upcalled all the way up the hierarchy from your setUp? e.g. Call super(%s, self).setUp() from your setUp().' % (sys.modules[self.__class__.__module__].__file__, self.__class__.__name__))
    return ret