from __future__ import annotations # isort:skip
import hashlib
import json
from os.path import splitext
from pathlib import Path
from sys import stdout
from typing import TYPE_CHECKING, Any, TextIO
from urllib.parse import urljoin
from urllib.request import urlopen
def _bokeh_dir(create: bool=False) -> Path:
    bokeh_dir = Path('~').expanduser() / '.bokeh'
    if not bokeh_dir.exists():
        if not create:
            return bokeh_dir
        print(f'Creating {bokeh_dir} directory')
        try:
            bokeh_dir.mkdir()
        except OSError:
            raise RuntimeError(f'could not create bokeh config directory at {bokeh_dir}')
    elif not bokeh_dir.is_dir():
        raise RuntimeError(f'{bokeh_dir} exists but is not a directory')
    return bokeh_dir