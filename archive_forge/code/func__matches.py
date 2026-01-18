from typing import Any, List, Sequence, Tuple, TypeVar
from hamcrest import (
from hamcrest.core.base_matcher import BaseMatcher
from hamcrest.core.core.allof import AllOf
from hamcrest.core.description import Description
from hamcrest.core.matcher import Matcher
from typing_extensions import Protocol
from twisted.python.failure import Failure
def _matches(self, item: Sequence[T]) -> bool:
    """
        Determine whether every element of the sequence is matched.
        """
    for elem in item:
        if not self.elementMatcher.matches(elem):
            return False
    return True