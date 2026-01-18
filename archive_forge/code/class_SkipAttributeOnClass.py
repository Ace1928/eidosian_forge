from unittest import skipIf
from twisted.trial.unittest import TestCase
class SkipAttributeOnClass(TestCase):
    """
    All tests should be skipped because skip attribute is set on
    this class.
    """
    skip = "'skip' attribute set on this class, so skip all tests"

    def test_one(self) -> None:
        raise Exception('Test should skip and never reach here')

    def test_two(self) -> None:
        raise Exception('Test should skip and never reach here')