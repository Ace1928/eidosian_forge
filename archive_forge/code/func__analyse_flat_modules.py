import itertools
import os
from fnmatch import fnmatchcase
from glob import glob
from pathlib import Path
from typing import (
import _distutils_hack.override  # noqa: F401
from distutils import log
from distutils.util import convert_path
def _analyse_flat_modules(self) -> bool:
    self.dist.py_modules = FlatLayoutModuleFinder.find(self._root_dir)
    log.debug(f'discovered py_modules -- {self.dist.py_modules}')
    self._ensure_no_accidental_inclusion(self.dist.py_modules, 'modules')
    return bool(self.dist.py_modules)