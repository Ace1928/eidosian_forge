from __future__ import annotations
import logging # isort:skip
from typing import Any, TypeVar, Union
from ...util.deprecation import deprecated
from .bases import (
from .primitive import Float, Int
from .singletons import Intrinsic, Undefined
class Byte(Interval[int]):
    """ Accept integral byte values (0-255).

    Example:

        .. code-block:: python

            >>> class ByteModel(HasProps):
            ...     prop = Byte(default=0)
            ...

            >>> m = ByteModel()

            >>> m.prop = 255

            >>> m.prop = 256  # ValueError !!

            >>> m.prop = 10.3 # ValueError !!

    """

    def __init__(self, default: Init[int]=0, help: str | None=None) -> None:
        super().__init__(Int, 0, 255, default=default, help=help)