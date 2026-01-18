from __future__ import annotations
from ..mesonlib import MesonException, OptionKey
from .. import mlog
from pathlib import Path
import typing as T
def _flags_to_list(raw: str) -> T.List[str]:
    res = []
    curr = ''
    escape = False
    in_string = False
    for i in raw:
        if escape:
            if i not in ['"', "'"]:
                curr += '\\'
            curr += i
            escape = False
        elif i == '\\':
            escape = True
        elif i in {'"', "'"}:
            in_string = not in_string
        elif i in {' ', '\n'}:
            if in_string:
                curr += i
            else:
                res += [curr]
                curr = ''
        else:
            curr += i
    res += [curr]
    res = [r for r in res if len(r) > 0]
    return res