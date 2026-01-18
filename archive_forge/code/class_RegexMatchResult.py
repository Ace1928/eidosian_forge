import dataclasses
import re
import warnings
from typing import (
@dataclasses.dataclass()
class RegexMatchResult(object):
    """
	The :class:`RegexMatchResult` data class is used to return information about
	the matched regular expression.
	"""
    __slots__ = ('match',)
    match: MatchHint
    '\n\t*match* (:class:`re.Match`) is the regex match result.\n\t'