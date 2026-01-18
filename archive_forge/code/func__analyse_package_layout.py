import itertools
import os
from fnmatch import fnmatchcase
from glob import glob
from pathlib import Path
from typing import (
import _distutils_hack.override  # noqa: F401
from distutils import log
from distutils.util import convert_path
def _analyse_package_layout(self, ignore_ext_modules: bool) -> bool:
    if self._explicitly_specified(ignore_ext_modules):
        return True
    log.debug('No `packages` or `py_modules` configuration, performing automatic discovery.')
    return self._analyse_explicit_layout() or self._analyse_src_layout() or self._analyse_flat_layout()