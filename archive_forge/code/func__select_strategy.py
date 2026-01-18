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
def _select_strategy(self, name: str, tag: str, build_lib: _Path) -> 'EditableStrategy':
    """Decides which strategy to use to implement an editable installation."""
    build_name = f'__editable__.{name}-{tag}'
    project_dir = Path(self.project_dir)
    mode = _EditableMode.convert(self.mode)
    if mode is _EditableMode.STRICT:
        auxiliary_dir = _empty_dir(Path(self.project_dir, 'build', build_name))
        return _LinkTree(self.distribution, name, auxiliary_dir, build_lib)
    packages = _find_packages(self.distribution)
    has_simple_layout = _simple_layout(packages, self.package_dir, project_dir)
    is_compat_mode = mode is _EditableMode.COMPAT
    if set(self.package_dir) == {''} and has_simple_layout or is_compat_mode:
        src_dir = self.package_dir.get('', '.')
        return _StaticPth(self.distribution, name, [Path(project_dir, src_dir)])
    return _TopLevelFinder(self.distribution, name)