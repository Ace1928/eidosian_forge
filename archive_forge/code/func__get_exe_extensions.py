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
def _get_exe_extensions() -> Sequence[str]:
    PATHEXT = os.environ.get('PATHEXT', None)
    if PATHEXT:
        return tuple((p.upper() for p in PATHEXT.split(os.pathsep)))
    elif os.name == 'nt':
        return ('.BAT', 'COM', '.EXE')
    else:
        return ()