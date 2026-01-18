import ast
import importlib
import os
import pathlib
import sys
from glob import iglob
from configparser import ConfigParser
from importlib.machinery import ModuleSpec
from itertools import chain
from typing import (
from pathlib import Path
from types import ModuleType
from distutils.errors import DistutilsOptionError
from .._path import same_path as _same_path
from ..warnings import SetuptoolsWarning
def _assert_local(filepath: _Path, root_dir: str):
    if Path(os.path.abspath(root_dir)) not in Path(os.path.abspath(filepath)).parents:
        msg = f'Cannot access {filepath!r} (or anything outside {root_dir!r})'
        raise DistutilsOptionError(msg)
    return True