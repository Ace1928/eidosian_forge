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
def _warn_if_mismatch(old: pathlib.Path, new: pathlib.Path, *, key: str) -> bool:
    if old == new:
        return False
    _warn_mismatched(old, new, key=key)
    return True