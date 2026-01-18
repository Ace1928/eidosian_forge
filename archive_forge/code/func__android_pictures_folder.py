from __future__ import annotations
import os
import re
import sys
from functools import lru_cache
from typing import cast
from .api import PlatformDirsABC
@lru_cache(maxsize=1)
def _android_pictures_folder() -> str:
    """:return: pictures folder for the Android OS"""
    try:
        from jnius import autoclass
        context = autoclass('android.content.Context')
        environment = autoclass('android.os.Environment')
        pictures_dir: str = context.getExternalFilesDir(environment.DIRECTORY_PICTURES).getAbsolutePath()
    except Exception:
        pictures_dir = '/storage/emulated/0/Pictures'
    return pictures_dir