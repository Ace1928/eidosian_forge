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
def _get_width(self):
    if self._width is None:
        self._width = get_regexp_width(self.to_regexp())
    return self._width