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
def _ensure_dist_info(self):
    if self.dist_info_dir is None:
        dist_info = self.reinitialize_command('dist_info')
        dist_info.output_dir = self.dist_dir
        dist_info.ensure_finalized()
        dist_info.run()
        self.dist_info_dir = dist_info.dist_info_dir
    else:
        assert str(self.dist_info_dir).endswith('.dist-info')
        assert Path(self.dist_info_dir, 'METADATA').exists()