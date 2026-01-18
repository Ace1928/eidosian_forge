import re
from functools import lru_cache
from typing import Callable, List
from ._cell_widths import CELL_WIDTHS
Break text in to equal (cell) length strings, returning the characters in reverse
    order