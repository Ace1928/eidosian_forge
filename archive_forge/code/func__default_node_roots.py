import asyncio
import enum
import json
import pathlib
import re
import shutil
import subprocess
import sys
from functools import lru_cache
from typing import (
from traitlets import Any as Any_
from traitlets import Instance
from traitlets import List as List_
from traitlets import Unicode, default
from traitlets.config import LoggingConfigurable
@default('node_roots')
def _default_node_roots(self):
    """get the "usual suspects" for where `node_modules` may be found

        - where this was launch (usually the same as NotebookApp.notebook_dir)
        - the JupyterLab staging folder (if available)
        - wherever conda puts it
        - wherever some other conventions put it
        """
    roots = [pathlib.Path.cwd()]
    try:
        from jupyterlab import commands
        roots += [pathlib.Path(commands.get_app_dir()) / 'staging']
    except ImportError:
        pass
    roots += [pathlib.Path(sys.prefix) / 'lib']
    roots += [pathlib.Path(sys.prefix)]
    npm = shutil.which('npm')
    if npm:
        prefix = self._npm_prefix(npm)
        if prefix:
            roots += [pathlib.Path(prefix) / 'lib', pathlib.Path(prefix)]
    return roots