import itertools
import os
from fnmatch import fnmatchcase
from glob import glob
from pathlib import Path
from typing import (
import _distutils_hack.override  # noqa: F401
from distutils import log
from distutils.util import convert_path
def _analyse_src_layout(self) -> bool:
    """Try to find all packages or modules under the ``src`` directory
        (or anything pointed by ``package_dir[""]``).

        The "src-layout" is relatively safe for automatic discovery.
        We assume that everything within is meant to be included in the
        distribution.

        If ``package_dir[""]`` is not given, but the ``src`` directory exists,
        this function will set ``package_dir[""] = "src"``.
        """
    package_dir = self._package_dir
    src_dir = os.path.join(self._root_dir, package_dir.get('', 'src'))
    if not os.path.isdir(src_dir):
        return False
    log.debug(f'`src-layout` detected -- analysing {src_dir}')
    package_dir.setdefault('', os.path.basename(src_dir))
    self.dist.package_dir = package_dir
    self.dist.packages = PEP420PackageFinder.find(src_dir)
    self.dist.py_modules = ModuleFinder.find(src_dir)
    log.debug(f'discovered packages -- {self.dist.packages}')
    log.debug(f'discovered py_modules -- {self.dist.py_modules}')
    return True