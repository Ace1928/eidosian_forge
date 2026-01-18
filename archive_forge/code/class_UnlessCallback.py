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
class UnlessCallback:

    def __init__(self, scanner):
        self.scanner = scanner

    def __call__(self, t):
        res = self.scanner.match(t.value, 0)
        if res:
            _value, t.type = res
        return t