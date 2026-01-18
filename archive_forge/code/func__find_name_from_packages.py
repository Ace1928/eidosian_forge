import itertools
import os
from fnmatch import fnmatchcase
from glob import glob
from pathlib import Path
from typing import (
import _distutils_hack.override  # noqa: F401
from distutils import log
from distutils.util import convert_path
def _find_name_from_packages(self) -> Optional[str]:
    """Try to find the root package that is not a PEP 420 namespace"""
    if not self.dist.packages:
        return None
    packages = remove_stubs(sorted(self.dist.packages, key=len))
    package_dir = self.dist.package_dir or {}
    parent_pkg = find_parent_package(packages, package_dir, self._root_dir)
    if parent_pkg:
        log.debug(f'Common parent package detected, name: {parent_pkg}')
        return parent_pkg
    log.warn('No parent package detected, impossible to derive `name`')
    return None