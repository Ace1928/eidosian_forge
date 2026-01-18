import gc
import importlib.util
import multiprocessing
import os
import platform
import socket
import sys
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from io import StringIO
from platform import system
from typing import (
import numpy as np
import pytest
from scipy import sparse
import xgboost as xgb
from xgboost.core import ArrayLike
from xgboost.sklearn import SklObjective
from xgboost.testing.data import (
from hypothesis import strategies
from hypothesis.extra.numpy import arrays
class DirectoryExcursion:
    """Change directory.  Change back and optionally cleaning up the directory when
    exit.

    """

    def __init__(self, path: os.PathLike, cleanup: bool=False):
        self.path = path
        self.curdir = os.path.normpath(os.path.abspath(os.path.curdir))
        self.cleanup = cleanup
        self.files: Set[str] = set()

    def __enter__(self) -> None:
        os.chdir(self.path)
        if self.cleanup:
            self.files = {os.path.join(root, f) for root, subdir, files in os.walk(os.path.expanduser(self.path)) for f in files}

    def __exit__(self, *args: Any) -> None:
        os.chdir(self.curdir)
        if self.cleanup:
            files = {os.path.join(root, f) for root, subdir, files in os.walk(os.path.expanduser(self.path)) for f in files}
            diff = files.difference(self.files)
            for f in diff:
                os.remove(f)