import functools
import logging
import os
import pathlib
import sys
import sysconfig
from typing import Any, Dict, Generator, Optional, Tuple
from pip._internal.models.scheme import SCHEME_KEYS, Scheme
from pip._internal.utils.compat import WINDOWS
from pip._internal.utils.deprecation import deprecated
from pip._internal.utils.virtualenv import running_under_virtualenv
from . import _sysconfig
from .base import (
@functools.lru_cache(maxsize=None)
def _looks_like_msys2_mingw_scheme() -> bool:
    """MSYS2 patches distutils and sysconfig to use a UNIX-like scheme.

    However, MSYS2 incorrectly patches sysconfig ``nt`` scheme. The fix is
    likely going to be included in their 3.10 release, so we ignore the warning.
    See msys2/MINGW-packages#9319.

    MSYS2 MINGW's patch uses lowercase ``"lib"`` instead of the usual uppercase,
    and is missing the final ``"site-packages"``.
    """
    paths = sysconfig.get_paths('nt', expand=False)
    return all(('Lib' not in p and 'lib' in p and (not p.endswith('site-packages')) for p in (paths[key] for key in ('platlib', 'purelib'))))