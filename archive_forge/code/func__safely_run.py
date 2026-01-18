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
def _safely_run(self, cmd_name: str):
    try:
        return self.run_command(cmd_name)
    except Exception:
        SetuptoolsDeprecationWarning.emit('Customization incompatible with editable install', f'\n                {traceback.format_exc()}\n\n                If you are seeing this warning it is very likely that a setuptools\n                plugin or customization overrides the `{cmd_name}` command, without\n                taking into consideration how editable installs run build steps\n                starting from setuptools v64.0.0.\n\n                Plugin authors and developers relying on custom build steps are\n                encouraged to update their `{cmd_name}` implementation considering the\n                information about editable installs in\n                https://setuptools.pypa.io/en/latest/userguide/extension.html.\n\n                For the time being `setuptools` will silence this error and ignore\n                the faulty command, but this behaviour will change in future versions.\n                ')