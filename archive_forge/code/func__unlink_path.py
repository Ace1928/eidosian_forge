import contextlib
import pathlib
from pathlib import Path
import re
import time
from typing import Union
from unittest import mock
def _unlink_path(path, missing_ok=False):
    cm = contextlib.nullcontext()
    if missing_ok:
        cm = contextlib.suppress(FileNotFoundError)
    with cm:
        path.unlink()