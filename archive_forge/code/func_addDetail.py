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
def addDetail(self, name, content_object):
    """Add a detail to be reported with this test's outcome.

        For more details see pydoc testtools.TestResult.

        :param name: The name to give this detail.
        :param content_object: The content object for this detail. See
            testtools.content for more detail.
        """
    if self.__details is None:
        self.__details = {}
    self.__details[name] = content_object