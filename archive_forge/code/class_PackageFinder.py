import itertools
import os
from fnmatch import fnmatchcase
from glob import glob
from pathlib import Path
from typing import (
import _distutils_hack.override  # noqa: F401
from distutils import log
from distutils.util import convert_path
class PackageFinder(_Finder):
    """
    Generate a list of all Python packages found within a directory
    """
    ALWAYS_EXCLUDE = ('ez_setup', '*__pycache__')

    @classmethod
    def _find_iter(cls, where: _Path, exclude: _Filter, include: _Filter) -> StrIter:
        """
        All the packages found in 'where' that pass the 'include' filter, but
        not the 'exclude' filter.
        """
        for root, dirs, files in os.walk(str(where), followlinks=True):
            all_dirs = dirs[:]
            dirs[:] = []
            for dir in all_dirs:
                full_path = os.path.join(root, dir)
                rel_path = os.path.relpath(full_path, where)
                package = rel_path.replace(os.path.sep, '.')
                if '.' in dir or not cls._looks_like_package(full_path, package):
                    continue
                if include(package) and (not exclude(package)):
                    yield package
                if f'{package}*' in exclude or f'{package}.*' in exclude:
                    continue
                dirs.append(dir)

    @staticmethod
    def _looks_like_package(path: _Path, _package_name: str) -> bool:
        """Does a directory look like a package?"""
        return os.path.isfile(os.path.join(path, '__init__.py'))