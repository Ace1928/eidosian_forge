from __future__ import annotations
from contextlib import (
import re
from typing import (
import warnings
from pandas._typing import (
from pandas.util._exceptions import find_stack_level
def _build_option_description(k: str) -> str:
    """Builds a formatted description of a registered option and prints it"""
    o = _get_registered_option(k)
    d = _get_deprecated_option(k)
    s = f'{k} '
    if o.doc:
        s += '\n'.join(o.doc.strip().split('\n'))
    else:
        s += 'No description available.'
    if o:
        s += f'\n    [default: {o.defval}] [currently: {_get_option(k, True)}]'
    if d:
        rkey = d.rkey or ''
        s += '\n    (Deprecated'
        s += f', use `{rkey}` instead.'
        s += ')'
    return s