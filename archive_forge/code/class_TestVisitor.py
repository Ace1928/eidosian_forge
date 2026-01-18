import logging
import unittest
import weakref
from typing import Dict, List
from .. import pyutils
class TestVisitor:
    """A visitor for Tests"""

    def visitSuite(self, aTestSuite):
        pass

    def visitCase(self, aTestCase):
        pass