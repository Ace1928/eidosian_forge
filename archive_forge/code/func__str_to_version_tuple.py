import dataclasses
import re
from dataclasses import dataclass
from functools import total_ordering
from typing import Optional, Union
def _str_to_version_tuple(version_str):
    """Return the tuple (major, minor, patch) version extracted from the str."""
    res = _VERSION_REG.match(version_str)
    if not res:
        raise ValueError(f"Invalid version '{version_str}'. Format should be x.y.z with {{x,y,z}} being digits.")
    return tuple((int(v) for v in [res.group('major'), res.group('minor'), res.group('patch')]))