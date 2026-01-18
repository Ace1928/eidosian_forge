import re
import sys
from functools import lru_cache
from typing import Final, List, Match, Pattern
from black._width_table import WIDTH_TABLE
from blib2to3.pytree import Leaf
@lru_cache(maxsize=64)
def _cached_compile(pattern: str) -> Pattern[str]:
    return re.compile(pattern)