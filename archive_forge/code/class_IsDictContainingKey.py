from typing import Any, Hashable, Mapping, TypeVar, Union
from hamcrest.core.base_matcher import BaseMatcher
from hamcrest.core.description import Description
from hamcrest.core.helpers.hasmethod import hasmethod
from hamcrest.core.helpers.wrap_matcher import wrap_matcher
from hamcrest.core.matcher import Matcher
class IsDictContainingKey(BaseMatcher[Mapping[K, Any]]):

    def __init__(self, key_matcher: Matcher[K]) -> None:
        self.key_matcher = key_matcher

    def _matches(self, item: Mapping[K, Any]) -> bool:
        if hasmethod(item, 'keys'):
            for key in item.keys():
                if self.key_matcher.matches(key):
                    return True
        return False

    def describe_to(self, description: Description) -> None:
        description.append_text('a dictionary containing key ').append_description_of(self.key_matcher)