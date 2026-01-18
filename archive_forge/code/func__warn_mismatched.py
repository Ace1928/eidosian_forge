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
def _warn_mismatched(old: pathlib.Path, new: pathlib.Path, *, key: str) -> None:
    issue_url = 'https://github.com/pypa/pip/issues/10151'
    message = 'Value for %s does not match. Please report this to <%s>\ndistutils: %s\nsysconfig: %s'
    logger.log(_MISMATCH_LEVEL, message, key, issue_url, old, new)