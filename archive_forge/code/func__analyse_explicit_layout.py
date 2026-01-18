import itertools
import os
from fnmatch import fnmatchcase
from glob import glob
from pathlib import Path
from typing import (
import _distutils_hack.override  # noqa: F401
from distutils import log
from distutils.util import convert_path
def _analyse_explicit_layout(self) -> bool:
    """The user can explicitly give a package layout via ``package_dir``"""
    package_dir = self._package_dir.copy()
    package_dir.pop('', None)
    root_dir = self._root_dir
    if not package_dir:
        return False
    log.debug(f'`explicit-layout` detected -- analysing {package_dir}')
    pkgs = chain_iter((_find_packages_within(pkg, os.path.join(root_dir, parent_dir)) for pkg, parent_dir in package_dir.items()))
    self.dist.packages = list(pkgs)
    log.debug(f'discovered packages -- {self.dist.packages}')
    return True