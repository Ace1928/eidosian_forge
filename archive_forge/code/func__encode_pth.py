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
def _encode_pth(content: str) -> bytes:
    """.pth files are always read with 'locale' encoding, the recommendation
    from the cpython core developers is to write them as ``open(path, "w")``
    and ignore warnings (see python/cpython#77102, pypa/setuptools#3937).
    This function tries to simulate this behaviour without having to create an
    actual file, in a way that supports a range of active Python versions.
    (There seems to be some variety in the way different version of Python handle
    ``encoding=None``, not all of them use ``locale.getpreferredencoding(False)``).
    """
    encoding = 'locale' if sys.version_info >= (3, 10) else None
    with io.BytesIO() as buffer:
        wrapper = io.TextIOWrapper(buffer, encoding)
        wrapper.write(content)
        wrapper.flush()
        buffer.seek(0)
        return buffer.read()