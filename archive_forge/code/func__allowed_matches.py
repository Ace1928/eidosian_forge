import warnings
from enum import Enum
from typing import List, Optional
from typing_extensions import Literal
@classmethod
def _allowed_matches(cls, source: str) -> List[str]:
    keys, vals = ([], [])
    for enum_key, enum_val in cls.__members__.items():
        keys.append(enum_key)
        vals.append(enum_val.value)
    if source == 'key':
        return keys
    if source == 'value':
        return vals
    return keys + vals