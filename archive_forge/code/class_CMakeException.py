from __future__ import annotations
from ..mesonlib import MesonException, OptionKey
from .. import mlog
from pathlib import Path
import typing as T
class CMakeException(MesonException):
    pass