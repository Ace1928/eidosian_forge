from typing import MutableSequence, Optional, Sequence, TypeVar, Union, cast
from hamcrest.core.base_matcher import BaseMatcher
from hamcrest.core.description import Description
from hamcrest.core.helpers.wrap_matcher import wrap_matcher
from hamcrest.core.matcher import Matcher
class IsSequenceContainingInAnyOrder(BaseMatcher[Sequence[T]]):

    def __init__(self, matchers: Sequence[Matcher[T]]) -> None:
        self.matchers = matchers

    def matches(self, item: Sequence[T], mismatch_description: Optional[Description]=None) -> bool:
        try:
            sequence = list(item)
            matchsequence = MatchInAnyOrder(self.matchers, mismatch_description)
            for element in sequence:
                if not matchsequence.matches(element):
                    return False
            return matchsequence.isfinished(sequence)
        except TypeError:
            if mismatch_description:
                super(IsSequenceContainingInAnyOrder, self).describe_mismatch(item, mismatch_description)
            return False

    def describe_mismatch(self, item: Sequence[T], mismatch_description: Description) -> None:
        self.matches(item, mismatch_description)

    def describe_to(self, description: Description) -> None:
        description.append_text('a sequence over ').append_list('[', ', ', ']', self.matchers).append_text(' in any order')