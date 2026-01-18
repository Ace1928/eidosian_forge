from __future__ import annotations
import contextlib
import datetime
import errno
import hashlib
import importlib
import importlib.util
import inspect
import locale
import os
import os.path
import re
import sys
import types
from types import ModuleType
from typing import (
from coverage import env
from coverage.exceptions import CoverageException
from coverage.types import TArc
from coverage.exceptions import *   # pylint: disable=wildcard-import
def _human_key(s: str) -> tuple[list[str | int], str]:
    """Turn a string into a list of string and number chunks.

    "z23a" -> (["z", 23, "a"], "z23a")

    The original string is appended as a last value to ensure the
    key is unique enough so that "x1y" and "x001y" can be distinguished.
    """

    def tryint(s: str) -> str | int:
        """If `s` is a number, return an int, else `s` unchanged."""
        try:
            return int(s)
        except ValueError:
            return s
    return ([tryint(c) for c in re.split('(\\d+)', s)], s)