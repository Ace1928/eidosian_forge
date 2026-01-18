from __future__ import annotations
import os
import sys
from configparser import ConfigParser
from pathlib import Path
from .api import PlatformDirsABC
def _get_user_dirs_folder(key: str) -> str | None:
    """Return directory from user-dirs.dirs config file. See https://freedesktop.org/wiki/Software/xdg-user-dirs/."""
    user_dirs_config_path = Path(Unix().user_config_dir) / 'user-dirs.dirs'
    if user_dirs_config_path.exists():
        parser = ConfigParser()
        with user_dirs_config_path.open() as stream:
            parser.read_string(f'[top]\n{stream.read()}')
        if key not in parser['top']:
            return None
        path = parser['top'][key].strip('"')
        return path.replace('$HOME', os.path.expanduser('~'))
    return None