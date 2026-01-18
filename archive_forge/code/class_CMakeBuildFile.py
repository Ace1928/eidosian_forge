from __future__ import annotations
from ..mesonlib import MesonException, OptionKey
from .. import mlog
from pathlib import Path
import typing as T
class CMakeBuildFile:

    def __init__(self, file: Path, is_cmake: bool, is_temp: bool) -> None:
        self.file = file
        self.is_cmake = is_cmake
        self.is_temp = is_temp

    def __repr__(self) -> str:
        return f'<{self.__class__.__name__}: {self.file}; cmake={self.is_cmake}; temp={self.is_temp}>'