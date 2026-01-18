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
class _StaticPth:

    def __init__(self, dist: Distribution, name: str, path_entries: List[Path]):
        self.dist = dist
        self.name = name
        self.path_entries = path_entries

    def __call__(self, wheel: 'WheelFile', files: List[str], mapping: Dict[str, str]):
        entries = '\n'.join((str(p.resolve()) for p in self.path_entries))
        contents = _encode_pth(f'{entries}\n')
        wheel.writestr(f'__editable__.{self.name}.pth', contents)

    def __enter__(self):
        msg = f'\n        Editable install will be performed using .pth file to extend `sys.path` with:\n        {list(map(os.fspath, self.path_entries))!r}\n        '
        _logger.warning(msg + _LENIENT_WARNING)
        return self

    def __exit__(self, _exc_type, _exc_value, _traceback):
        ...