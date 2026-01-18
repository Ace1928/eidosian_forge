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
def _future_update(self, type: Optional[str]=None, value: Optional[Any]=None) -> 'Token':
    return Token.new_borrow_pos(type if type is not None else self.type, value if value is not None else self.value, self)