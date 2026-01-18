from testtools.matchers import *
from . import CapturedCall, TestCase, TestCaseWithTransport
from .matchers import *
class FakeUnlockable:
    """Something that can be unlocked."""

    def unlock(self):
        pass