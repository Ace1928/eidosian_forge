import itertools
import os
from fnmatch import fnmatchcase
from glob import glob
from pathlib import Path
from typing import (
import _distutils_hack.override  # noqa: F401
from distutils import log
from distutils.util import convert_path
class _Finder:
    """Base class that exposes functionality for module/package finders"""
    ALWAYS_EXCLUDE: Tuple[str, ...] = ()
    DEFAULT_EXCLUDE: Tuple[str, ...] = ()

    @classmethod
    def find(cls, where: _Path='.', exclude: Iterable[str]=(), include: Iterable[str]=('*',)) -> List[str]:
        """Return a list of all Python items (packages or modules, depending on
        the finder implementation) found within directory 'where'.

        'where' is the root directory which will be searched.
        It should be supplied as a "cross-platform" (i.e. URL-style) path;
        it will be converted to the appropriate local path syntax.

        'exclude' is a sequence of names to exclude; '*' can be used
        as a wildcard in the names.
        When finding packages, 'foo.*' will exclude all subpackages of 'foo'
        (but not 'foo' itself).

        'include' is a sequence of names to include.
        If it's specified, only the named items will be included.
        If it's not specified, all found items will be included.
        'include' can contain shell style wildcard patterns just like
        'exclude'.
        """
        exclude = exclude or cls.DEFAULT_EXCLUDE
        return list(cls._find_iter(convert_path(str(where)), _Filter(*cls.ALWAYS_EXCLUDE, *exclude), _Filter(*include)))

    @classmethod
    def _find_iter(cls, where: _Path, exclude: _Filter, include: _Filter) -> StrIter:
        raise NotImplementedError