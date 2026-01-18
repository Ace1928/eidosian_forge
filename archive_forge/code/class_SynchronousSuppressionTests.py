import unittest as pyunit
from twisted.python.reflect import namedAny
from twisted.trial import unittest
from twisted.trial.test import suppression
class SynchronousSuppressionTests(SuppressionMixin, unittest.SynchronousTestCase):
    """
    @see: L{twisted.trial.test.test_tests}
    """
    TestSetUpSuppression = namedAny('twisted.trial.test.suppression.SynchronousTestSetUpSuppression')
    TestTearDownSuppression = namedAny('twisted.trial.test.suppression.SynchronousTestTearDownSuppression')
    TestSuppression = namedAny('twisted.trial.test.suppression.SynchronousTestSuppression')
    TestSuppression2 = namedAny('twisted.trial.test.suppression.SynchronousTestSuppression2')