import re
from typing import Any
from ..helpers import NORMALIZE_PATTERN, collapse_white_spaces
from .atomic_types import AnyAtomicType
class XsdToken(NormalizedString):
    name = 'token'
    pattern = re.compile('^[\\S\\xa0]*(?: [\\S\\xa0]+)*$')

    def __new__(cls, value: Any) -> 'XsdToken':
        if not isinstance(value, str):
            value = str(value)
        else:
            value = collapse_white_spaces(value)
        match = cls.pattern.match(value)
        if match is None:
            raise ValueError('invalid value {!r} for xs:{}'.format(value, cls.name))
        return super(NormalizedString, cls).__new__(cls, value)