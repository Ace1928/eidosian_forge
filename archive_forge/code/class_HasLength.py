from collections.abc import Sized
from typing import Union
from hamcrest.core.base_matcher import BaseMatcher
from hamcrest.core.description import Description
from hamcrest.core.helpers.hasmethod import hasmethod
from hamcrest.core.helpers.wrap_matcher import wrap_matcher
from hamcrest.core.matcher import Matcher
class HasLength(BaseMatcher[Sized]):

    def __init__(self, len_matcher: Matcher[int]) -> None:
        self.len_matcher = len_matcher

    def _matches(self, item: Sized) -> bool:
        if not hasmethod(item, '__len__'):
            return False
        return self.len_matcher.matches(len(item))

    def describe_mismatch(self, item: Sized, mismatch_description: Description) -> None:
        super(HasLength, self).describe_mismatch(item, mismatch_description)
        if hasmethod(item, '__len__'):
            mismatch_description.append_text(' with length of ').append_description_of(len(item))

    def describe_to(self, description: Description) -> None:
        description.append_text('an object with length of ').append_description_of(self.len_matcher)