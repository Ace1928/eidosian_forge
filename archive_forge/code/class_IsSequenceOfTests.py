from typing import Callable, Sequence, Tuple, Type
from hamcrest import anything, assert_that, contains, contains_string, equal_to, not_
from hamcrest.core.matcher import Matcher
from hamcrest.core.string_description import StringDescription
from hypothesis import given
from hypothesis.strategies import (
from twisted.python.failure import Failure
from twisted.trial.unittest import SynchronousTestCase
from .matchers import HasSum, IsSequenceOf, S, isFailure, similarFrame
class IsSequenceOfTests(SynchronousTestCase):
    """
    Tests for L{IsSequenceOf}.
    """
    sequences = lists(booleans())

    @given(integers(min_value=0, max_value=1000))
    def test_matches(self, numItems: int) -> None:
        """
        L{IsSequenceOf} matches a sequence if all of the elements are
        matched by the parameterized matcher.

        :param numItems: The length of a sequence to try to match.
        """
        seq = [True] * numItems
        matcher = IsSequenceOf(equal_to(True))
        actualDescription = StringDescription()
        assert_that(matcher.matches(seq, actualDescription), equal_to(True))
        assert_that(str(actualDescription), equal_to(''))

    @given(integers(min_value=0, max_value=1000), integers(min_value=0, max_value=1000))
    def test_mismatches(self, numBefore: int, numAfter: int) -> None:
        """
        L{IsSequenceOf} does not match a sequence if any of the elements
        are not matched by the parameterized matcher.

        :param numBefore: In the sequence to try to match, the number of
            elements expected to match before an expected mismatch.

        :param numAfter: In the sequence to try to match, the number of
            elements expected expected to match after an expected mismatch.
        """
        seq = [True] * numBefore + [False] + [True] * numAfter
        matcher = IsSequenceOf(equal_to(True))
        actualDescription = StringDescription()
        assert_that(matcher.matches(seq, actualDescription), equal_to(False))
        actualStr = str(actualDescription)
        assert_that(actualStr, contains_string('a sequence containing only'))
        assert_that(actualStr, contains_string(f'not sequence with element #{numBefore}'))