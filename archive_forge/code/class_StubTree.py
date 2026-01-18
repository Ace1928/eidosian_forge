from testtools.matchers import *
from . import CapturedCall, TestCase, TestCaseWithTransport
from .matchers import *
class StubTree:
    """Stubg for testing."""

    def __init__(self, lock_status):
        self._is_locked = lock_status

    def __str__(self):
        return 'I am da tree'

    def is_locked(self):
        return self._is_locked