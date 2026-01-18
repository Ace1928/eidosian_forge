from enum import Enum
from functools import lru_cache
from typing import (
class UnsetType(Enum):
    """Sentintel object - used for defaults where None
    may be a valid (non-default) value.

    This class is an enum so it may used as a Literal
    type annotation, e.g Literal[UNSET]."""
    UNSET = 'UNSET'

    def __bool__(self) -> Literal[False]:
        return False