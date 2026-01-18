from __future__ import annotations
from ..mesonlib import MesonException, OptionKey
from .. import mlog
from pathlib import Path
import typing as T
class CMakeInclude:

    def __init__(self, path: Path, isSystem: bool=False):
        self.path = path
        self.isSystem = isSystem

    def __repr__(self) -> str:
        return f'<CMakeInclude: {self.path} -- isSystem = {self.isSystem}>'