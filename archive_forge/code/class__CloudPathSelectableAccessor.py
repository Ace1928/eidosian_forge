import abc
from collections import defaultdict
import collections.abc
from contextlib import contextmanager
import os
from pathlib import (  # type: ignore
import shutil
import sys
from typing import (
from urllib.parse import urlparse
from warnings import warn
from cloudpathlib.enums import FileCacheMode
from . import anypath
from .exceptions import (
class _CloudPathSelectableAccessor:

    def __init__(self, scandir_func: Callable) -> None:
        self.scandir = scandir_func