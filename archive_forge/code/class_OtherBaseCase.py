import unittest
import testtools
import testresources
from testresources.tests import ResultWithResourceExtensions
class OtherBaseCase(unittest.TestCase):
    tearDownCalled = False

    def tearDown(self):
        self.tearDownCalled = True
        super(OtherBaseCase, self).setUp()