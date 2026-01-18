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
def _log_context(*, user: bool=False, home: Optional[str]=None, root: Optional[str]=None, prefix: Optional[str]=None) -> None:
    parts = ['Additional context:', 'user = %r', 'home = %r', 'root = %r', 'prefix = %r']
    logger.log(_MISMATCH_LEVEL, '\n'.join(parts), user, home, root, prefix)