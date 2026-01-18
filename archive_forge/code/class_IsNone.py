from typing import Any, Optional
from hamcrest.core.base_matcher import BaseMatcher
from hamcrest.core.description import Description
from hamcrest.core.matcher import Matcher
from .isnot import is_not
class IsNone(BaseMatcher[Optional[Any]]):

    def _matches(self, item: Any) -> bool:
        return item is None

    def describe_to(self, description: Description) -> None:
        description.append_text('None')