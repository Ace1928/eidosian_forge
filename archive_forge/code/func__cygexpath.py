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
def _cygexpath(drive: Optional[str], path: str) -> str:
    if osp.isabs(path) and (not drive):
        p = path
    else:
        p = path and osp.normpath(osp.expandvars(osp.expanduser(path)))
        if osp.isabs(p):
            if drive:
                p = path
            else:
                p = cygpath(p)
        elif drive:
            p = '/proc/cygdrive/%s/%s' % (drive.lower(), p)
    p_str = str(p)
    return p_str.replace('\\', '/')