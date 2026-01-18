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
def _find_namespaces(packages: List[str], pkg_roots: Dict[str, str]) -> Iterator[Tuple[str, List[str]]]:
    for pkg in packages:
        path = find_package_path(pkg, pkg_roots, '')
        if Path(path).exists() and (not Path(path, '__init__.py').exists()):
            yield (pkg, [path])