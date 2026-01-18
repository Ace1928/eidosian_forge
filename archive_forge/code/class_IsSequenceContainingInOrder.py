import warnings
from typing import Optional, Sequence, TypeVar, Union
from hamcrest.core.base_matcher import BaseMatcher
from hamcrest.core.description import Description
from hamcrest.core.helpers.wrap_matcher import wrap_matcher
from hamcrest.core.matcher import Matcher
class IsSequenceContainingInOrder(BaseMatcher[Sequence[T]]):

    def __init__(self, matchers: Sequence[Matcher[T]]) -> None:
        self.matchers = matchers

    def matches(self, item: Sequence[T], mismatch_description: Optional[Description]=None) -> bool:
        try:
            matchsequence = MatchingInOrder(self.matchers, mismatch_description)
            for element in item:
                if not matchsequence.matches(element):
                    return False
            return matchsequence.isfinished()
        except TypeError:
            if mismatch_description:
                super(IsSequenceContainingInOrder, self).describe_mismatch(item, mismatch_description)
            return False

    def describe_mismatch(self, item: Sequence[T], mismatch_description: Description) -> None:
        self.matches(item, mismatch_description)

    def describe_to(self, description: Description) -> None:
        description.append_text('a sequence containing ').append_list('[', ', ', ']', self.matchers)