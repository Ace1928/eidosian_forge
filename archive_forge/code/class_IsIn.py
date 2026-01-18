from typing import Sequence, TypeVar
from hamcrest.core.base_matcher import BaseMatcher
from hamcrest.core.description import Description
from hamcrest.core.matcher import Matcher
class IsIn(BaseMatcher[T]):

    def __init__(self, sequence: Sequence[T]) -> None:
        self.sequence = sequence

    def _matches(self, item: T) -> bool:
        return item in self.sequence

    def describe_to(self, description: Description) -> None:
        description.append_text('one of ').append_list('(', ', ', ')', self.sequence)