import contextlib
import warnings
from collections import defaultdict
from enum import IntEnum
from typing import (
class TypeDescriptor(NamedTuple):
    """
    Class for config data manipulating of exact type.

    Parameters
    ----------
    decode : callable
        Callable to decode config value from the raw data.
    normalize : callable
        Callable to bring different config value variations to
        the single form.
    verify : callable
        Callable to check that config value satisfies given config
        type requirements.
    help : str
        Class description string.
    """
    decode: Callable[[str], object]
    normalize: Callable[[object], object]
    verify: Callable[[object], bool]
    help: str