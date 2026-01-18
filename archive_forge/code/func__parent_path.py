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
def _parent_path(pkg, pkg_path):
    """Infer the parent path containing a package, that if added to ``sys.path`` would
    allow importing that package.
    When ``pkg`` is directly mapped into a directory with a different name, return its
    own path.
    >>> _parent_path("a", "src/a")
    'src'
    >>> _parent_path("b", "src/c")
    'src/c'
    """
    parent = pkg_path[:-len(pkg)] if pkg_path.endswith(pkg) else pkg_path
    return parent.rstrip('/' + os.sep)