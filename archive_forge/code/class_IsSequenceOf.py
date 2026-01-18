from typing import Any, List, Sequence, Tuple, TypeVar
from hamcrest import (
from hamcrest.core.base_matcher import BaseMatcher
from hamcrest.core.core.allof import AllOf
from hamcrest.core.description import Description
from hamcrest.core.matcher import Matcher
from typing_extensions import Protocol
from twisted.python.failure import Failure
class IsSequenceOf(BaseMatcher[Sequence[T]]):
    """
    Match a sequence where every element is matched by another matcher.

    :ivar elementMatcher: The matcher which must match every element of the
        sequence.
    """

    def __init__(self, elementMatcher: Matcher[T]) -> None:
        self.elementMatcher = elementMatcher

    def _matches(self, item: Sequence[T]) -> bool:
        """
        Determine whether every element of the sequence is matched.
        """
        for elem in item:
            if not self.elementMatcher.matches(elem):
                return False
        return True

    def describe_mismatch(self, item: Sequence[T], description: Description) -> None:
        """
        Describe the mismatch.
        """
        for idx, elem in enumerate(item):
            if not self.elementMatcher.matches(elem):
                description.append_description_of(self)
                description.append_text(f'not sequence with element #{idx} {elem!r}')

    def describe_to(self, description: Description) -> None:
        """
        Describe this matcher for error messages.
        """
        description.append_text('a sequence containing only ')
        description.append_description_of(self.elementMatcher)
        description.append_text(', ')