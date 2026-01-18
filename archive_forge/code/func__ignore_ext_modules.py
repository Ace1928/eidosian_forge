import itertools
import os
from fnmatch import fnmatchcase
from glob import glob
from pathlib import Path
from typing import (
import _distutils_hack.override  # noqa: F401
from distutils import log
from distutils.util import convert_path
def _ignore_ext_modules(self):
    """Internal API to disregard ext_modules.

        Normally auto-discovery would not be triggered if ``ext_modules`` are set
        (this is done for backward compatibility with existing packages relying on
        ``setup.py`` or ``setup.cfg``). However, ``setuptools`` can call this function
        to ignore given ``ext_modules`` and proceed with the auto-discovery if
        ``packages`` and ``py_modules`` are not given (e.g. when using pyproject.toml
        metadata).
        """
    self._skip_ext_modules = True