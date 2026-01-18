import itertools
import os
from fnmatch import fnmatchcase
from glob import glob
from pathlib import Path
from typing import (
import _distutils_hack.override  # noqa: F401
from distutils import log
from distutils.util import convert_path
def _find_name_single_package_or_module(self) -> Optional[str]:
    """Exactly one module or package"""
    for field in ('packages', 'py_modules'):
        items = getattr(self.dist, field, None) or []
        if items and len(items) == 1:
            log.debug(f'Single module/package detected, name: {items[0]}')
            return items[0]
    return None