from . import idnadata
import bisect
import unicodedata
import re
from typing import Union, Optional
from .intranges import intranges_contain
def _combining_class(cp: int) -> int:
    v = unicodedata.combining(chr(cp))
    if v == 0:
        if not unicodedata.name(chr(cp)):
            raise ValueError('Unknown character in unicodedata')
    return v