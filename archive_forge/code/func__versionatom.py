import platform
import re
import sys
import typing
def _versionatom(s: str) -> int:
    if s.isdigit():
        return int(s)
    match = RE_NUM.match(s)
    return int(match.groups()[0]) if match else 0