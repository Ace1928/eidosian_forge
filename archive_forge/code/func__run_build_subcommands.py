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
def _run_build_subcommands(self):
    """
        Issue #3501 indicates that some plugins/customizations might rely on:

        1. ``build_py`` not running
        2. ``build_py`` always copying files to ``build_lib``

        However both these assumptions may be false in editable_wheel.
        This method implements a temporary workaround to support the ecosystem
        while the implementations catch up.
        """
    build: Command = self.get_finalized_command('build')
    for name in build.get_sub_commands():
        cmd = self.get_finalized_command(name)
        if name == 'build_py' and type(cmd) != build_py_cls:
            self._safely_run(name)
        else:
            self.run_command(name)