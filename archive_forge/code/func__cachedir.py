from __future__ import annotations
import inspect
import os
from pathlib import Path
import re
import shutil
import sys
import tempfile
from .base import TestBase
from .. import config
from ..assertions import eq_
from ... import util
def _cachedir(self):
    mypy_path = ''
    with tempfile.TemporaryDirectory() as cachedir:
        with open(Path(cachedir) / 'sqla_mypy_config.cfg', 'w') as config_file:
            config_file.write(f'\n                    [mypy]\n\n                    plugins = sqlalchemy.ext.mypy.plugin\n\n                    show_error_codes = True\n\n                    {mypy_path}\n                    disable_error_code = no-untyped-call\n\n                    [mypy-sqlalchemy.*]\n                    ignore_errors = True\n\n                    ')
        with open(Path(cachedir) / 'plain_mypy_config.cfg', 'w') as config_file:
            config_file.write(f'\n                    [mypy]\n\n                    show_error_codes = True\n\n                    {mypy_path}\n                    disable_error_code = var-annotated,no-untyped-call\n                    [mypy-sqlalchemy.*]\n                    ignore_errors = True\n\n                    ')
        yield cachedir