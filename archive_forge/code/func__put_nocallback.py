import contextlib
import warnings
from collections import defaultdict
from enum import IntEnum
from typing import (
@classmethod
def _put_nocallback(cls, value: Any) -> Any:
    """
        Set config value without executing callbacks.

        Parameters
        ----------
        value : Any
            Config value to set.

        Returns
        -------
        Any
            Replaced (old) config value.
        """
    if not _TYPE_PARAMS[cls.type].verify(value):
        raise ValueError(f'Unsupported value: {value}')
    value = _TYPE_PARAMS[cls.type].normalize(value)
    oldvalue, cls._value = (cls.get(), value)
    return oldvalue