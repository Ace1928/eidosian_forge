from __future__ import annotations
from .. import mparser
from .exceptions import InvalidCode, InvalidArguments
from .helpers import flatten, resolve_second_level_holders
from .operator import MesonOperator
from ..mesonlib import HoldableObject, MesonBugException
import textwrap
import typing as T
from abc import ABCMeta
from contextlib import AbstractContextManager
def _throw_comp_exception(self, other: TYPE_var, opt_type: str) -> T.NoReturn:
    raise InvalidArguments(textwrap.dedent(f'\n                Trying to compare values of different types ({self.display_name()}, {type(other).__name__}) using {opt_type}.\n                This was deprecated and undefined behavior previously and is as of 0.60.0 a hard error.\n            '))