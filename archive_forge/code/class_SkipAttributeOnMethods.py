from unittest import skipIf
from twisted.trial.unittest import TestCase
class SkipAttributeOnMethods(TestCase):
    """
    Only methods where @skipIf decorator is used should be skipped.
    """

    def test_one(self) -> None:
        raise Exception('Should never reach here')
    test_one.skip = 'skip test, skip attribute set on method'

    def test_shouldNotSkip(self) -> None:
        self.assertTrue(True, 'Test should run and not be skipped')