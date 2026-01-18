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
def _create_wheel_file(self, bdist_wheel):
    from wheel.wheelfile import WheelFile
    dist_info = self.get_finalized_command('dist_info')
    dist_name = dist_info.name
    tag = '-'.join(bdist_wheel.get_tag())
    build_tag = '0.editable'
    archive_name = f'{dist_name}-{build_tag}-{tag}.whl'
    wheel_path = Path(self.dist_dir, archive_name)
    if wheel_path.exists():
        wheel_path.unlink()
    unpacked_wheel = TemporaryDirectory(suffix=archive_name)
    build_lib = TemporaryDirectory(suffix='.build-lib')
    build_tmp = TemporaryDirectory(suffix='.build-temp')
    with unpacked_wheel as unpacked, build_lib as lib, build_tmp as tmp:
        unpacked_dist_info = Path(unpacked, Path(self.dist_info_dir).name)
        shutil.copytree(self.dist_info_dir, unpacked_dist_info)
        self._install_namespaces(unpacked, dist_name)
        files, mapping = self._run_build_commands(dist_name, unpacked, lib, tmp)
        strategy = self._select_strategy(dist_name, tag, lib)
        with strategy, WheelFile(wheel_path, 'w') as wheel_obj:
            strategy(wheel_obj, files, mapping)
            wheel_obj.write_files(unpacked)
    return wheel_path