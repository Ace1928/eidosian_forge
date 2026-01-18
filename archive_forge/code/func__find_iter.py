import itertools
import os
from fnmatch import fnmatchcase
from glob import glob
from pathlib import Path
from typing import (
import _distutils_hack.override  # noqa: F401
from distutils import log
from distutils.util import convert_path
@classmethod
def _find_iter(cls, where: _Path, exclude: _Filter, include: _Filter) -> StrIter:
    for file in glob(os.path.join(where, '*.py')):
        module, _ext = os.path.splitext(os.path.basename(file))
        if not cls._looks_like_module(module):
            continue
        if include(module) and (not exclude(module)):
            yield module