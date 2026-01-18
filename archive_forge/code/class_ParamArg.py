import enum
import typing
from functools import lru_cache
from typing import Any, Dict, FrozenSet, Iterable, List, Tuple, Union
class ParamArg(enum.Enum):
    """Enum representing special args to ``load()``.

    FULL: used to request all attributes
    DEFAULT: used to request the default attribute
    """
    FULL = 'full'
    DEFAULT = 'default'

    @classmethod
    @lru_cache(maxsize=1)
    def values(cls) -> FrozenSet[str]:
        """Returns all values."""
        return frozenset((arg.value for arg in cls))

    @classmethod
    def is_arg(cls, val: Union['ParamArg', str]) -> bool:
        """Returns true if ``val`` is a ``ParamArg``, or one
        of its values."""
        return isinstance(val, ParamArg) or (isinstance(val, str) and val in cls.values())

    def __str__(self) -> str:
        return self.value