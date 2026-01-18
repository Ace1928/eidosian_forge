from __future__ import annotations
import importlib.util
import os
import re
import shutil
from collections import defaultdict
from collections.abc import Iterator
from typing import TYPE_CHECKING
import pytest
from numpy.typing.mypy_plugin import _EXTENDED_PRECISION_LIST
def _strip_filename(msg: str) -> tuple[int, str]:
    """Strip the filename and line number from a mypy message."""
    _, tail = os.path.splitdrive(msg)
    _, lineno, msg = tail.split(':', 2)
    return (int(lineno), msg.strip())