from __future__ import annotations
from contextlib import (
import re
from typing import (
import warnings
from pandas._typing import (
from pandas.util._exceptions import find_stack_level
def _get_single_key(pat: str, silent: bool) -> str:
    keys = _select_options(pat)
    if len(keys) == 0:
        if not silent:
            _warn_if_deprecated(pat)
        raise OptionError(f'No such keys(s): {repr(pat)}')
    if len(keys) > 1:
        raise OptionError('Pattern matched multiple keys')
    key = keys[0]
    if not silent:
        _warn_if_deprecated(key)
    key = _translate_key(key)
    return key