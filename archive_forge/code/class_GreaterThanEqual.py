import collections
import logging
from unittest import mock
import fixtures
from oslotest import base
from testtools import compat
from testtools import matchers
from testtools import testcase
from taskflow import exceptions
from taskflow.tests import fixtures as taskflow_fixtures
from taskflow.tests import utils
from taskflow.utils import misc
class GreaterThanEqual(object):
    """Matches if the item is geq than the matchers reference object."""

    def __init__(self, source):
        self.source = source

    def match(self, other):
        if other >= self.source:
            return None
        return matchers.Mismatch('%s was not >= %s' % (other, self.source))