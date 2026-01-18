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
def _looks_like_deb_system_dist_packages(value: str) -> bool:
    """Check if the value is Debian's APT-controlled dist-packages.

    Debian's ``distutils.sysconfig.get_python_lib()`` implementation returns the
    default package path controlled by APT, but does not patch ``sysconfig`` to
    do the same. This is similar to the bug worked around in ``get_scheme()``,
    but here the default is ``deb_system`` instead of ``unix_local``. Ultimately
    we can't do anything about this Debian bug, and this detection allows us to
    skip the warning when needed.
    """
    if not _looks_like_debian_scheme():
        return False
    if value == '/usr/lib/python3/dist-packages':
        return True
    return False