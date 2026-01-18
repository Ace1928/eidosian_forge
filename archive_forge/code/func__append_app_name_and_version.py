from __future__ import annotations
import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING
def _append_app_name_and_version(self, *base: str) -> str:
    params = list(base[1:])
    if self.appname:
        params.append(self.appname)
        if self.version:
            params.append(self.version)
    path = os.path.join(base[0], *params)
    self._optionally_create_directory(path)
    return path