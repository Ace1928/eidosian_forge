from __future__ import annotations
from contextlib import (
import re
from typing import (
import warnings
from pandas._typing import (
from pandas.util._exceptions import find_stack_level
def _translate_key(key: str) -> str:
    """
    if key id deprecated and a replacement key defined, will return the
    replacement key, otherwise returns `key` as - is
    """
    d = _get_deprecated_option(key)
    if d:
        return d.rkey or key
    else:
        return key