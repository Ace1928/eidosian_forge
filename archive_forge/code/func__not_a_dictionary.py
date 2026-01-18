from typing import Any, Hashable, Mapping, Optional, TypeVar, Union, overload
from hamcrest.core.base_matcher import BaseMatcher
from hamcrest.core.description import Description
from hamcrest.core.helpers.wrap_matcher import wrap_matcher
from hamcrest.core.matcher import Matcher
def _not_a_dictionary(self, item: Mapping[K, V], mismatch_description: Optional[Description]) -> bool:
    if mismatch_description:
        mismatch_description.append_description_of(item).append_text(' is not a mapping object')
    return False