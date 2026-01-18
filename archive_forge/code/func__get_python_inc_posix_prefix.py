import os
import re
import sys
import sysconfig
import pathlib
from .errors import DistutilsPlatformError
from . import py39compat
from ._functools import pass_none
def _get_python_inc_posix_prefix(prefix):
    implementation = 'pypy' if IS_PYPY else 'python'
    python_dir = implementation + get_python_version() + build_flags
    return os.path.join(prefix, 'include', python_dir)