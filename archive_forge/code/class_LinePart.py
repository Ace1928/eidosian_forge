import re
from dataclasses import dataclass
from itertools import chain
from typing import Optional, Tuple
from .lazyre import LazyReCompile
@dataclass
class LinePart:
    start: int
    stop: int
    word: str