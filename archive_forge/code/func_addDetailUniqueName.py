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
def addDetailUniqueName(self, name, content_object):
    """Add a detail to the test, but ensure it's name is unique.

        This method checks whether ``name`` conflicts with a detail that has
        already been added to the test. If it does, it will modify ``name`` to
        avoid the conflict.

        For more details see pydoc testtools.TestResult.

        :param name: The name to give this detail.
        :param content_object: The content object for this detail. See
            testtools.content for more detail.
        """
    existing_details = self.getDetails()
    full_name = name
    suffix = 1
    while full_name in existing_details:
        full_name = '%s-%d' % (name, suffix)
        suffix += 1
    self.addDetail(full_name, content_object)