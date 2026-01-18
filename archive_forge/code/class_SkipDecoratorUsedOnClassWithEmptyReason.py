from unittest import skipIf
from twisted.trial.unittest import TestCase
@skipIf(True, '')
class SkipDecoratorUsedOnClassWithEmptyReason(TestCase):
    """
    All tests should be skipped because @skipIf decorator is used on
    this class, even if the reason is an empty string
    """

    def test_shouldNeverRun_1(self) -> None:
        raise Exception('Test should skip and never reach here')

    def test_shouldNeverRun_2(self) -> None:
        raise Exception('Test should skip and never reach here')