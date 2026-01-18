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
def _configure_build(self, name: str, unpacked_wheel: _Path, build_lib: _Path, tmp_dir: _Path):
    """Configure commands to behave in the following ways:

        - Build commands can write to ``build_lib`` if they really want to...
          (but this folder is expected to be ignored and modules are expected to live
          in the project directory...)
        - Binary extensions should be built in-place (editable_mode = True)
        - Data/header/script files are not part of the "editable" specification
          so they are written directly to the unpacked_wheel directory.
        """
    dist = self.distribution
    wheel = str(unpacked_wheel)
    build_lib = str(build_lib)
    data = str(Path(unpacked_wheel, f'{name}.data', 'data'))
    headers = str(Path(unpacked_wheel, f'{name}.data', 'headers'))
    scripts = str(Path(unpacked_wheel, f'{name}.data', 'scripts'))
    egg_info = dist.reinitialize_command('egg_info', reinit_subcommands=True)
    egg_info.egg_base = str(tmp_dir)
    egg_info.ignore_egg_info_in_manifest = True
    build = dist.reinitialize_command('build', reinit_subcommands=True)
    install = dist.reinitialize_command('install', reinit_subcommands=True)
    build.build_platlib = build.build_purelib = build.build_lib = build_lib
    install.install_purelib = install.install_platlib = install.install_lib = wheel
    install.install_scripts = build.build_scripts = scripts
    install.install_headers = headers
    install.install_data = data
    install_scripts = dist.get_command_obj('install_scripts')
    install_scripts.no_ep = True
    build.build_temp = str(tmp_dir)
    build_py = dist.get_command_obj('build_py')
    build_py.compile = False
    build_py.existing_egg_info_dir = self._find_egg_info_dir()
    self._set_editable_mode()
    build.ensure_finalized()
    install.ensure_finalized()