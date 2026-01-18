from __future__ import annotations
from fontTools.misc.textTools import byteord, tostr
import re
from bisect import bisect_right
from typing import Literal, TypeVar, overload
from . import Blocks, Scripts, ScriptExtensions, OTTags
def _normalize_property_name(string):
    """Remove case, strip space, '-' and '_' for loose matching."""
    return _normalize_re.sub('', string).lower()