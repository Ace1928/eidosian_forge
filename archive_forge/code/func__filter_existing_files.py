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
def _filter_existing_files(filepaths: Iterable[_Path]) -> Iterator[_Path]:
    for path in filepaths:
        if os.path.isfile(path):
            yield path
        else:
            SetuptoolsWarning.emit(f'File {path!r} cannot be found')