from typing import Hashable, Mapping, MutableMapping, TypeVar, Union
from hamcrest.core.base_matcher import BaseMatcher
from hamcrest.core.description import Description
from hamcrest.core.helpers.hasmethod import hasmethod
from hamcrest.core.helpers.wrap_matcher import wrap_matcher
from hamcrest.core.matcher import Matcher
def _matching_keys(self, item) -> MutableMapping[K, V]:
    key_matches: MutableMapping[K, V] = {}
    if hasmethod(item, 'items'):
        for key, value in item.items():
            if self.key_matcher.matches(key):
                key_matches[key] = value
    return key_matches