from __future__ import annotations
from dataclasses import dataclass
import os
import abc
import typing as T
class MesonBugException(MesonException):
    """Exceptions thrown when there is a clear Meson bug that should be reported"""

    def __init__(self, msg: str, file: T.Optional[str]=None, lineno: T.Optional[int]=None, colno: T.Optional[int]=None):
        super().__init__(msg + '\n\n    This is a Meson bug and should be reported!', file=file, lineno=lineno, colno=colno)