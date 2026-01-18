from typing import Any, Hashable, Mapping, Optional, TypeVar, Union, overload
from hamcrest.core.base_matcher import BaseMatcher
from hamcrest.core.description import Description
from hamcrest.core.helpers.wrap_matcher import wrap_matcher
from hamcrest.core.matcher import Matcher
class IsDictContainingEntries(BaseMatcher[Mapping[K, V]]):

    def __init__(self, value_matchers) -> None:
        self.value_matchers = sorted(value_matchers.items())

    def _not_a_dictionary(self, item: Mapping[K, V], mismatch_description: Optional[Description]) -> bool:
        if mismatch_description:
            mismatch_description.append_description_of(item).append_text(' is not a mapping object')
        return False

    def matches(self, item: Mapping[K, V], mismatch_description: Optional[Description]=None) -> bool:
        for key, value_matcher in self.value_matchers:
            try:
                if key not in item:
                    if mismatch_description:
                        mismatch_description.append_text('no ').append_description_of(key).append_text(' key in ').append_description_of(item)
                    return False
            except TypeError:
                return self._not_a_dictionary(item, mismatch_description)
            try:
                actual_value = item[key]
            except TypeError:
                return self._not_a_dictionary(item, mismatch_description)
            if not value_matcher.matches(actual_value):
                if mismatch_description:
                    mismatch_description.append_text('value for ').append_description_of(key).append_text(' ')
                    value_matcher.describe_mismatch(actual_value, mismatch_description)
                return False
        return True

    def describe_mismatch(self, item: Mapping[K, V], mismatch_description: Description) -> None:
        self.matches(item, mismatch_description)

    def describe_keyvalue(self, index: K, value: V, description: Description) -> None:
        """Describes key-value pair at given index."""
        description.append_description_of(index).append_text(': ').append_description_of(value)

    def describe_to(self, description: Description) -> None:
        description.append_text('a dictionary containing {')
        first = True
        for key, value in self.value_matchers:
            if not first:
                description.append_text(', ')
            self.describe_keyvalue(key, value, description)
            first = False
        description.append_text('}')