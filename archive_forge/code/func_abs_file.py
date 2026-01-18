from __future__ import annotations
import hashlib
import ntpath
import os
import os.path
import posixpath
import re
import sys
from typing import Callable, Iterable
from coverage import env
from coverage.exceptions import ConfigError
from coverage.misc import human_sorted, isolate_module, join_regex
def abs_file(path: str) -> str:
    """Return the absolute normalized form of `path`."""
    return actual_path(os.path.abspath(os.path.realpath(path)))