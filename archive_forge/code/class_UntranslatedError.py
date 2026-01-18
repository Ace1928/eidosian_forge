from twisted.internet.defer import (
from twisted.trial.unittest import SynchronousTestCase, TestCase
class UntranslatedError(Exception):
    """
    Untranslated exception type when testing an exception translation.
    """