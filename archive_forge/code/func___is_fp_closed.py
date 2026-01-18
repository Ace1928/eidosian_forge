from __future__ import annotations
import mmap
from tempfile import NamedTemporaryFile
from typing import TYPE_CHECKING, Any, Callable
def __is_fp_closed(self) -> bool:
    try:
        return self.__fp.fp is None
    except AttributeError:
        pass
    try:
        closed: bool = self.__fp.closed
        return closed
    except AttributeError:
        pass
    return False