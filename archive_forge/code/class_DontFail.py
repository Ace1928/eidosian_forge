from twisted.internet.defer import (
from twisted.trial.unittest import SynchronousTestCase, TestCase
class DontFail(Exception):
    """
    Sample exception type.
    """

    def __init__(self, actual):
        Exception.__init__(self)
        self.actualValue = actual