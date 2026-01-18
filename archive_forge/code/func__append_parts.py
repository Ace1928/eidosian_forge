from __future__ import annotations
import ctypes
import os
import sys
from functools import lru_cache
from typing import TYPE_CHECKING
from .api import PlatformDirsABC
def _append_parts(self, path: str, *, opinion_value: str | None=None) -> str:
    params = []
    if self.appname:
        if self.appauthor is not False:
            author = self.appauthor or self.appname
            params.append(author)
        params.append(self.appname)
        if opinion_value is not None and self.opinion:
            params.append(opinion_value)
        if self.version:
            params.append(self.version)
    path = os.path.join(path, *params)
    self._optionally_create_directory(path)
    return path