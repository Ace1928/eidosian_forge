import re
import typing as t
from ast import literal_eval
from collections import deque
from sys import intern
from ._identifier import pattern as name_re
from .exceptions import TemplateSyntaxError
from .utils import LRUCache
def _normalize_newlines(self, value: str) -> str:
    """Replace all newlines with the configured sequence in strings
        and template data.
        """
    return newline_re.sub(self.newline_sequence, value)