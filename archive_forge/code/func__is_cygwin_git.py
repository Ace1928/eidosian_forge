import sys
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
import time
from urllib.parse import urlsplit, urlunsplit
import warnings
from gitdb.util import (
from typing import (
from git.types import (
def _is_cygwin_git(git_executable: str) -> bool:
    is_cygwin = _is_cygwin_cache.get(git_executable)
    if is_cygwin is None:
        is_cygwin = False
        try:
            git_dir = osp.dirname(git_executable)
            if not git_dir:
                res = py_where(git_executable)
                git_dir = osp.dirname(res[0]) if res else ''
            uname_cmd = osp.join(git_dir, 'uname')
            process = subprocess.Popen([uname_cmd], stdout=subprocess.PIPE, universal_newlines=True)
            uname_out, _ = process.communicate()
            is_cygwin = 'CYGWIN' in uname_out
        except Exception as ex:
            _logger.debug('Failed checking if running in CYGWIN due to: %r', ex)
        _is_cygwin_cache[git_executable] = is_cygwin
    return is_cygwin