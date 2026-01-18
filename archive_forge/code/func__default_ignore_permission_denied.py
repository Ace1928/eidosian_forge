import logging
import os
import sys
import warnings
from enum import IntEnum
from pathlib import Path
from typing import TYPE_CHECKING, AsyncGenerator, Callable, Generator, Optional, Set, Tuple, Union
import anyio
from ._rust_notify import RustNotify
from .filters import DefaultFilter
def _default_ignore_permission_denied(ignore_permission_denied: Optional[bool]) -> bool:
    if ignore_permission_denied is not None:
        return ignore_permission_denied
    env_var = os.getenv('WATCHFILES_IGNORE_PERMISSION_DENIED')
    return bool(env_var)