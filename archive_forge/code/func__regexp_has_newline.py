from abc import abstractmethod, ABC
import re
from contextlib import suppress
from typing import (
from types import ModuleType
import warnings
from .utils import classify, get_regexp_width, Serialize, logger
from .exceptions import UnexpectedCharacters, LexError, UnexpectedToken
from .grammar import TOKEN_DEFAULT_PRIORITY
from copy import copy
def _regexp_has_newline(r: str):
    """Expressions that may indicate newlines in a regexp:
        - newlines (\\n)
        - escaped newline (\\\\n)
        - anything but ([^...])
        - any-char (.) when the flag (?s) exists
        - spaces (\\s)
    """
    return '\n' in r or '\\n' in r or '\\s' in r or ('[^' in r) or ('(?s' in r and '.' in r)