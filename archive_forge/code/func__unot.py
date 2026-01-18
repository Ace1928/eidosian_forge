from . import idnadata
import bisect
import unicodedata
import re
from typing import Union, Optional
from .intranges import intranges_contain
def _unot(s: int) -> str:
    return 'U+{:04X}'.format(s)