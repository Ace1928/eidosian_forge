import collections.abc
import dataclasses
import functools
import inspect
import io
import itertools
import tokenize
from typing import Callable, Dict, Generic, Hashable, List, Optional, Type, TypeVar
import docstring_parser
from typing_extensions import get_origin, is_typeddict
from . import _resolver, _strings, _unsafe_cache
@dataclasses.dataclass(frozen=True)
class _Token:
    token_type: int
    content: str
    logical_line: int
    actual_line: int