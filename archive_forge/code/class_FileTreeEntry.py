from __future__ import annotations
import functools, uuid, os, operator
import typing as T
from . import backends
from .. import build
from .. import mesonlib
from .. import mlog
from ..mesonlib import MesonBugException, MesonException, OptionKey
class FileTreeEntry:

    def __init__(self) -> None:
        self.subdirs: T.Dict[str, FileTreeEntry] = {}
        self.targets: T.List[build.BuildTarget] = []