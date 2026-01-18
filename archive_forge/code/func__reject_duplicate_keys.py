import math
import re
from typing import (
import unicodedata
from .parser import Parser
def _reject_duplicate_keys(pairs, dictify):
    keys = set()
    for key, _ in pairs:
        if key in keys:
            raise ValueError(f'Duplicate key "{key}" found in object')
        keys.add(key)
    return dictify(pairs)