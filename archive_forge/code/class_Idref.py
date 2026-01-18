import re
from typing import Any
from ..helpers import NORMALIZE_PATTERN, collapse_white_spaces
from .atomic_types import AnyAtomicType
class Idref(NCName):
    name = 'IDREF'