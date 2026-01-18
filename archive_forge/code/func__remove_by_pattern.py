from __future__ import annotations
import fnmatch
import os
import subprocess
import sys
import threading
import time
import typing as t
from itertools import chain
from pathlib import PurePath
from ._internal import _log
def _remove_by_pattern(paths: set[str], exclude_patterns: set[str]) -> None:
    for pattern in exclude_patterns:
        paths.difference_update(fnmatch.filter(paths, pattern))