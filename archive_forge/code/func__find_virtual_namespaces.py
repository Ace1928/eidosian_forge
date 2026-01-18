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
def _find_virtual_namespaces(pkg_roots: Dict[str, str]) -> Iterator[str]:
    """By carefully designing ``package_dir``, it is possible to implement the logical
    structure of PEP 420 in a package without the corresponding directories.

    Moreover a parent package can be purposefully/accidentally skipped in the discovery
    phase (e.g. ``find_packages(include=["mypkg.*"])``, when ``mypkg.foo`` is included
    by ``mypkg`` itself is not).
    We consider this case to also be a virtual namespace (ignoring the original
    directory) to emulate a non-editable installation.

    This function will try to find these kinds of namespaces.
    """
    for pkg in pkg_roots:
        if '.' not in pkg:
            continue
        parts = pkg.split('.')
        for i in range(len(parts) - 1, 0, -1):
            partial_name = '.'.join(parts[:i])
            path = Path(find_package_path(partial_name, pkg_roots, ''))
            if not path.exists() or partial_name not in pkg_roots:
                yield partial_name