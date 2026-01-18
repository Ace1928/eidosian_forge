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
class _NamespaceInstaller(namespaces.Installer):

    def __init__(self, distribution, installation_dir, editable_name, src_root):
        self.distribution = distribution
        self.src_root = src_root
        self.installation_dir = installation_dir
        self.editable_name = editable_name
        self.outputs = []
        self.dry_run = False

    def _get_nspkg_file(self):
        """Installation target."""
        return os.path.join(self.installation_dir, self.editable_name + self.nspkg_ext)

    def _get_root(self):
        """Where the modules/packages should be loaded from."""
        return repr(str(self.src_root))