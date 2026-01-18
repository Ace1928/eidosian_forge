import re
from typing import Any
from ..helpers import NORMALIZE_PATTERN, collapse_white_spaces
from .atomic_types import AnyAtomicType
class NMToken(XsdToken):
    name = 'NMTOKEN'
    pattern = re.compile('^[\\w.\\-:\\u00B7\\u0300-\\u036F\\u203F\\u2040]+$')