from testtools.helpers import try_import
from testtools import TestCase
class NeedsTwistedTestCase(TestCase):

    def setUp(self):
        super().setUp()
        if defer is None:
            self.skipTest('Need Twisted to run')