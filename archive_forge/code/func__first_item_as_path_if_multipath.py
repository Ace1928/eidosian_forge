from __future__ import annotations
import os
import sys
from configparser import ConfigParser
from pathlib import Path
from .api import PlatformDirsABC
def _first_item_as_path_if_multipath(self, directory: str) -> Path:
    if self.multipath:
        directory = directory.split(os.pathsep)[0]
    return Path(directory)