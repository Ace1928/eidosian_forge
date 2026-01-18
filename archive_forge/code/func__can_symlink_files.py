import logging
import io
import os
import shutil
import sys
import traceback
from contextlib import suppress
from enum import Enum
from inspect import cleandoc
from itertools import chain, starmap
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import (
from .. import (
from ..discovery import find_package_path
from ..dist import Distribution
from ..warnings import (
from .build_py import build_py as build_py_cls
import sys
from importlib.machinery import ModuleSpec, PathFinder
from importlib.machinery import all_suffixes as module_suffixes
from importlib.util import spec_from_file_location
from itertools import chain
from pathlib import Path
def _can_symlink_files(base_dir: Path) -> bool:
    with TemporaryDirectory(dir=str(base_dir.resolve())) as tmp:
        path1, path2 = (Path(tmp, 'file1.txt'), Path(tmp, 'file2.txt'))
        path1.write_text('file1', encoding='utf-8')
        with suppress(AttributeError, NotImplementedError, OSError):
            os.symlink(path1, path2)
            if path2.is_symlink() and path2.read_text(encoding='utf-8') == 'file1':
                return True
        try:
            os.link(path1, path2)
        except Exception as ex:
            msg = 'File system does not seem to support either symlinks or hard links. Strict editable installs require one of them to be supported.'
            raise LinksNotSupported(msg) from ex
        return False