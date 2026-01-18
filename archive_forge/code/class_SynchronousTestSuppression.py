import warnings
from twisted.trial import unittest, util
class SynchronousTestSuppression(SuppressionMixin, unittest.SynchronousTestCase):
    pass