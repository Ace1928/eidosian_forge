import logging
import io
import os
import shutil
import sys
import traceback
from contextlib import suppress
from enum import Enum
from inspect import cleandoc
from itertools import chain, starmap
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import (
from .. import (
from ..discovery import find_package_path
from ..dist import Distribution
from ..warnings import (
from .build_py import build_py as build_py_cls
import sys
from importlib.machinery import ModuleSpec, PathFinder
from importlib.machinery import all_suffixes as module_suffixes
from importlib.util import spec_from_file_location
from itertools import chain
from pathlib import Path
def _install_namespaces(self, installation_dir, pth_prefix):
    dist = self.distribution
    if not dist.namespace_packages:
        return
    src_root = Path(self.project_dir, self.package_dir.get('', '.')).resolve()
    installer = _NamespaceInstaller(dist, installation_dir, pth_prefix, src_root)
    installer.install_namespaces()