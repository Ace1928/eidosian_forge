import warnings
from twisted.trial import unittest, util
class SynchronousTestSuppression2(TestSuppression2Mixin, unittest.SynchronousTestCase):
    pass