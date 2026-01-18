from typing import Sequence, TypeVar, Union, cast
from hamcrest.core.base_matcher import BaseMatcher
from hamcrest.core.core.allof import all_of
from hamcrest.core.description import Description
from hamcrest.core.helpers.wrap_matcher import wrap_matcher
from hamcrest.core.matcher import Matcher
class IsSequenceContainingEvery(BaseMatcher[Sequence[T]]):

    def __init__(self, *element_matchers: Matcher[T]) -> None:
        delegates = [cast(Matcher[Sequence[T]], has_item(e)) for e in element_matchers]
        self.matcher: Matcher[Sequence[T]] = all_of(*delegates)

    def _matches(self, item: Sequence[T]) -> bool:
        try:
            return self.matcher.matches(list(item))
        except TypeError:
            return False

    def describe_mismatch(self, item: Sequence[T], mismatch_description: Description) -> None:
        self.matcher.describe_mismatch(item, mismatch_description)

    def describe_to(self, description: Description) -> None:
        self.matcher.describe_to(description)