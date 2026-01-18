import os
import io
import logging
import shutil
import stat
from pathlib import Path
from contextlib import contextmanager
from .. import __version__ as full_version
from ..utils import check_version, get_logger
def _recursive_chmod_directories(root, mode):
    """
    Recursively change the permissions on the child directories using a bitwise
    OR operation.
    """
    for item in root.iterdir():
        if item.is_dir():
            item.chmod(item.stat().st_mode | mode)
            _recursive_chmod_directories(item, mode)