import itertools
import os
from fnmatch import fnmatchcase
from glob import glob
from pathlib import Path
from typing import (
import _distutils_hack.override  # noqa: F401
from distutils import log
from distutils.util import convert_path
def _explicitly_specified(self, ignore_ext_modules: bool) -> bool:
    """``True`` if the user has specified some form of package/module listing"""
    ignore_ext_modules = ignore_ext_modules or self._skip_ext_modules
    ext_modules = not (self.dist.ext_modules is None or ignore_ext_modules)
    return self.dist.packages is not None or self.dist.py_modules is not None or ext_modules or (hasattr(self.dist, 'configuration') and self.dist.configuration)