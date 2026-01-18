from typing import Any
from hamcrest.core.helpers.wrap_matcher import wrap_matcher
from hamcrest.core.matcher import Matcher
from hamcrest.core.string_description import tostring
class EqualityWrapper(object):

    def __init__(self, matcher: Matcher) -> None:
        self.matcher = matcher

    def __eq__(self, obj: Any) -> bool:
        return self.matcher.matches(obj)

    def __str__(self) -> str:
        return repr(self)

    def __repr__(self) -> str:
        return tostring(self.matcher)