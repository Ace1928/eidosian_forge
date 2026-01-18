from __future__ import annotations
import logging # isort:skip
import hashlib
import json
import os
import re
import sys
from os.path import (
from pathlib import Path
from subprocess import PIPE, Popen
from typing import Any, Callable, Sequence
from ..core.has_props import HasProps
from ..settings import settings
from .strings import snakify
def _detect_nodejs() -> Path:
    nodejs_path = settings.nodejs_path()
    nodejs_paths = [nodejs_path] if nodejs_path is not None else ['nodejs', 'node']
    for nodejs_path in nodejs_paths:
        try:
            proc = Popen([nodejs_path, '--version'], stdout=PIPE, stderr=PIPE)
            stdout, _ = proc.communicate()
        except OSError:
            continue
        if proc.returncode != 0:
            continue
        match = re.match('^v(\\d+)\\.(\\d+)\\.(\\d+).*$', stdout.decode('utf-8'))
        if match is not None:
            version = tuple((int(v) for v in match.groups()))
            if version >= nodejs_min_version:
                return Path(nodejs_path)
    version_repr = '.'.join((str(x) for x in nodejs_min_version))
    raise RuntimeError(f'node.js v{version_repr} or higher is needed to allow compilation of custom models ' + '("conda install nodejs" or follow https://nodejs.org/en/download/)')