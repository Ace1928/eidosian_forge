from __future__ import annotations
import itertools
import re
from enum import Enum
from typing import Hashable, TypeVar
from prompt_toolkit.cache import SimpleCache
from .base import (
from .named_colors import NAMED_COLORS
def _expand_classname(classname: str) -> list[str]:
    """
    Split a single class name at the `.` operator, and build a list of classes.

    E.g. 'a.b.c' becomes ['a', 'a.b', 'a.b.c']
    """
    result = []
    parts = classname.split('.')
    for i in range(1, len(parts) + 1):
        result.append('.'.join(parts[:i]).lower())
    return result