import enum
import string
import unicodedata
from collections import defaultdict
import regex._regex as _regex
def _compile_firstset(info, fs):
    """Compiles the firstset for the pattern."""
    reverse = bool(info.flags & REVERSE)
    fs = _check_firstset(info, reverse, fs)
    if not fs:
        return []
    return fs.compile(reverse)