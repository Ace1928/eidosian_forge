from typing import Any, List, Sequence, Tuple, TypeVar
from hamcrest import (
from hamcrest.core.base_matcher import BaseMatcher
from hamcrest.core.core.allof import AllOf
from hamcrest.core.description import Description
from hamcrest.core.matcher import Matcher
from typing_extensions import Protocol
from twisted.python.failure import Failure
class HasSum(BaseMatcher[Sequence[S]]):
    """
    Match a sequence the elements of which sum to a value matched by
    another matcher.

    :ivar sumMatcher: The matcher which must match the sum.
    :ivar zero: The zero value for the matched type.
    """

    def __init__(self, sumMatcher: Matcher[S], zero: S) -> None:
        self.sumMatcher = sumMatcher
        self.zero = zero

    def _sum(self, sequence: Sequence[S]) -> S:
        if not sequence:
            return self.zero
        result = self.zero
        for elem in sequence:
            result = result + elem
        return result

    def _matches(self, item: Sequence[S]) -> bool:
        """
        Determine whether the sum of the sequence is matched.
        """
        s = self._sum(item)
        return self.sumMatcher.matches(s)

    def describe_mismatch(self, item: Sequence[S], description: Description) -> None:
        """
        Describe the mismatch.
        """
        s = self._sum(item)
        description.append_description_of(self)
        self.sumMatcher.describe_mismatch(s, description)
        return None

    def describe_to(self, description: Description) -> None:
        """
        Describe this matcher for error messages.
        """
        description.append_text('a sequence with sum ')
        description.append_description_of(self.sumMatcher)
        description.append_text(', ')