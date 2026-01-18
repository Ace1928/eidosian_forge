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
def _simple_layout(packages: Iterable[str], package_dir: Dict[str, str], project_dir: Path) -> bool:
    """Return ``True`` if:
    - all packages are contained by the same parent directory, **and**
    - all packages become importable if the parent directory is added to ``sys.path``.

    >>> _simple_layout(['a'], {"": "src"}, "/tmp/myproj")
    True
    >>> _simple_layout(['a', 'a.b'], {"": "src"}, "/tmp/myproj")
    True
    >>> _simple_layout(['a', 'a.b'], {}, "/tmp/myproj")
    True
    >>> _simple_layout(['a', 'a.a1', 'a.a1.a2', 'b'], {"": "src"}, "/tmp/myproj")
    True
    >>> _simple_layout(['a', 'a.a1', 'a.a1.a2', 'b'], {"a": "a", "b": "b"}, ".")
    True
    >>> _simple_layout(['a', 'a.a1', 'a.a1.a2', 'b'], {"a": "_a", "b": "_b"}, ".")
    False
    >>> _simple_layout(['a', 'a.a1', 'a.a1.a2', 'b'], {"a": "_a"}, "/tmp/myproj")
    False
    >>> _simple_layout(['a', 'a.a1', 'a.a1.a2', 'b'], {"a.a1.a2": "_a2"}, ".")
    False
    >>> _simple_layout(['a', 'a.b'], {"": "src", "a.b": "_ab"}, "/tmp/myproj")
    False
    >>> # Special cases, no packages yet:
    >>> _simple_layout([], {"": "src"}, "/tmp/myproj")
    True
    >>> _simple_layout([], {"a": "_a", "": "src"}, "/tmp/myproj")
    False
    """
    layout = {pkg: find_package_path(pkg, package_dir, project_dir) for pkg in packages}
    if not layout:
        return set(package_dir) in ({}, {''})
    parent = os.path.commonpath(starmap(_parent_path, layout.items()))
    return all((_path.same_path(Path(parent, *key.split('.')), value) for key, value in layout.items()))