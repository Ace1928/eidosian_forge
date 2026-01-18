from __future__ import annotations
import os
import re
import sys
from functools import lru_cache
from typing import cast
from .api import PlatformDirsABC
@lru_cache(maxsize=1)
def _android_videos_folder() -> str:
    """:return: videos folder for the Android OS"""
    try:
        from jnius import autoclass
        context = autoclass('android.content.Context')
        environment = autoclass('android.os.Environment')
        videos_dir: str = context.getExternalFilesDir(environment.DIRECTORY_DCIM).getAbsolutePath()
    except Exception:
        videos_dir = '/storage/emulated/0/DCIM/Camera'
    return videos_dir