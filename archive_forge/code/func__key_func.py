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
def _key_func(key: str) -> str:
    """Split at the first occurrence of the ``:`` character.

    Windows drive-letters (*e.g.* ``C:``) are ignored herein.
    """
    drive, tail = os.path.splitdrive(key)
    return os.path.join(drive, tail.split(':', 1)[0])