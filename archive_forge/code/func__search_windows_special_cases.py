from __future__ import annotations
import functools
import os
import shutil
import stat
import sys
import re
import typing as T
from pathlib import Path
from . import mesonlib
from . import mlog
from .mesonlib import MachineChoice, OrderedSet
def _search_windows_special_cases(self, name: str, command: str) -> T.List[T.Optional[str]]:
    """
        Lots of weird Windows quirks:
        1. PATH search for @name returns files with extensions from PATHEXT,
           but only self.windows_exts are executable without an interpreter.
        2. @name might be an absolute path to an executable, but without the
           extension. This works inside MinGW so people use it a lot.
        3. The script is specified without an extension, in which case we have
           to manually search in PATH.
        4. More special-casing for the shebang inside the script.
        """
    if command:
        name_ext = os.path.splitext(command)[1]
        if name_ext[1:].lower() in self.windows_exts:
            return [command]
        commands = self._shebang_to_cmd(command)
        if commands:
            return commands
        return [None]
    if os.path.isabs(name):
        for ext in self.windows_exts:
            command = f'{name}.{ext}'
            if os.path.exists(command):
                return [command]
    search_dirs = self._windows_sanitize_path(os.environ.get('PATH', '')).split(';')
    for search_dir in search_dirs:
        commands = self._search_dir(name, search_dir)
        if commands:
            return commands
    return [None]