import re
from typing import Any
from ..helpers import NORMALIZE_PATTERN, collapse_white_spaces
from .atomic_types import AnyAtomicType
class NormalizedString(str, AnyAtomicType):
    name = 'normalizedString'
    pattern = re.compile('^[^\t\r]*$')

    def __new__(cls, obj: Any) -> 'NormalizedString':
        try:
            return super().__new__(cls, NORMALIZE_PATTERN.sub(' ', obj))
        except TypeError:
            return super().__new__(cls, obj)