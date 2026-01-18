from unittest import skipIf
from twisted.trial.unittest import TestCase
@skipIf(True, 'Skip all tests when @skipIf is used on a class')
class SkipDecoratorUsedOnClass(TestCase):
    """
    All tests should be skipped because @skipIf decorator is used on
    this class.
    """

    def test_shouldNeverRun_1(self) -> None:
        raise Exception('Test should skip and never reach here')

    def test_shouldNeverRun_2(self) -> None:
        raise Exception('Test should skip and never reach here')