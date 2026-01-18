from __future__ import annotations
import os
import sys
from configparser import ConfigParser
from pathlib import Path
from .api import PlatformDirsABC
def _get_user_media_dir(env_var: str, fallback_tilde_path: str) -> str:
    media_dir = _get_user_dirs_folder(env_var)
    if media_dir is None:
        media_dir = os.environ.get(env_var, '').strip()
        if not media_dir:
            media_dir = os.path.expanduser(fallback_tilde_path)
    return media_dir