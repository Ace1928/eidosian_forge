from abc import abstractmethod
import contextlib
from functools import wraps
import getpass
import logging
import os
import os.path as osp
import pathlib
import platform
import re
import shutil
import stat
import subprocess
import sys
import time
from urllib.parse import urlsplit, urlunsplit
import warnings
from typing import (
from .types import (
from gitdb.util import (  # noqa: F401  # @IgnorePep8
class IterableClassWatcher(type):
    """Metaclass that issues :class:`DeprecationWarning` when :class:`git.util.Iterable`
    is subclassed."""

    def __init__(cls, name: str, bases: Tuple, clsdict: Dict) -> None:
        for base in bases:
            if type(base) is IterableClassWatcher:
                warnings.warn(f'GitPython Iterable subclassed by {name}. Iterable is deprecated due to naming clash since v3.1.18 and will be removed in 4.0.0. Use IterableObj instead.', DeprecationWarning, stacklevel=2)