from . import idnadata
import bisect
import unicodedata
import re
from typing import Union, Optional
from .intranges import intranges_contain
def _punycode(s: str) -> bytes:
    return s.encode('punycode')