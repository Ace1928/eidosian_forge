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
def _fix_abiflags(parts: Tuple[str]) -> Generator[str, None, None]:
    ldversion = sysconfig.get_config_var('LDVERSION')
    abiflags = getattr(sys, 'abiflags', None)
    if not ldversion or not abiflags or (not ldversion.endswith(abiflags)):
        yield from parts
        return
    for part in parts:
        if part.endswith(ldversion):
            part = part[:0 - len(abiflags)]
        yield part