from __future__ import annotations
import io
import re
from functools import partial
from pprint import pformat
from re import Match
from textwrap import fill
from typing import Any, Callable, Pattern
def abbrtask(S: str, max: int) -> str:
    """Abbreviate task name."""
    if S is None:
        return '???'
    if len(S) > max:
        module, _, cls = S.rpartition('.')
        module = abbr(module, max - len(cls) - 3, False)
        return module + '[.]' + cls
    return S