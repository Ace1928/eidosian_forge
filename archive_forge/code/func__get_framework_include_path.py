from __future__ import annotations
from .common import cmake_is_debug
from .. import mlog
from ..mesonlib import Version
from pathlib import Path
import re
import typing as T
def _get_framework_include_path(path: Path) -> T.Optional[str]:
    trials = ('Headers', 'Versions/Current/Headers', _get_framework_latest_version(path))
    for each in trials:
        trial = path / each
        if trial.is_dir():
            return trial.as_posix()
    return None