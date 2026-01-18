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
def _default_force_polling(force_polling: Optional[bool]) -> bool:
    """
    See docstring for `watch` above for details.

    See samuelcolvin/watchfiles#167 and samuelcolvin/watchfiles#187 for discussion and rationale.
    """
    if force_polling is not None:
        return force_polling
    env_var = os.getenv('WATCHFILES_FORCE_POLLING')
    if env_var:
        return env_var.lower() not in {'false', 'disable', 'disabled'}
    else:
        return _auto_force_polling()