import warnings
from typing import Optional, Sequence, TypeVar, Union
from hamcrest.core.base_matcher import BaseMatcher
from hamcrest.core.description import Description
from hamcrest.core.helpers.wrap_matcher import wrap_matcher
from hamcrest.core.matcher import Matcher
class MatchingInOrder(object):

    def __init__(self, matchers: Sequence[Matcher[T]], mismatch_description: Optional[Description]) -> None:
        self.matchers = matchers
        self.mismatch_description = mismatch_description
        self.next_match_index = 0

    def matches(self, item: T) -> bool:
        return self.isnotsurplus(item) and self.ismatched(item)

    def isfinished(self) -> bool:
        if self.next_match_index < len(self.matchers):
            if self.mismatch_description:
                self.mismatch_description.append_text('No item matched: ').append_description_of(self.matchers[self.next_match_index])
            return False
        return True

    def ismatched(self, item: T) -> bool:
        matcher = self.matchers[self.next_match_index]
        if not matcher.matches(item):
            if self.mismatch_description:
                self.mismatch_description.append_text('item ' + str(self.next_match_index) + ': ')
                matcher.describe_mismatch(item, self.mismatch_description)
            return False
        self.next_match_index += 1
        return True

    def isnotsurplus(self, item: T) -> bool:
        if len(self.matchers) <= self.next_match_index:
            if self.mismatch_description:
                self.mismatch_description.append_text('Not matched: ').append_description_of(item)
            return False
        return True