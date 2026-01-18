from decimal import Decimal
from math import fabs
from typing import Any, Union, overload
from hamcrest.core.base_matcher import BaseMatcher
from hamcrest.core.description import Description
from hamcrest.core.matcher import Matcher
class IsCloseTo(BaseMatcher[Number]):

    def __init__(self, value: Number, delta: Number) -> None:
        if not isnumeric(value):
            raise TypeError('IsCloseTo value must be numeric')
        if not isnumeric(delta):
            raise TypeError('IsCloseTo delta must be numeric')
        self.value = value
        self.delta = delta

    def _matches(self, item: Number) -> bool:
        if not isnumeric(item):
            return False
        return self._diff(item) <= self.delta

    def _diff(self, item: Number) -> float:
        return fabs(item - self.value)

    def describe_mismatch(self, item: Number, mismatch_description: Description) -> None:
        if not isnumeric(item):
            super(IsCloseTo, self).describe_mismatch(item, mismatch_description)
        else:
            actual_delta = self._diff(item)
            mismatch_description.append_description_of(item).append_text(' differed by ').append_description_of(actual_delta)

    def describe_to(self, description: Description) -> None:
        description.append_text('a numeric value within ').append_description_of(self.delta).append_text(' of ').append_description_of(self.value)