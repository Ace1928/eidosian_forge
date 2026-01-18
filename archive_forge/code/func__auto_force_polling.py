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
def _auto_force_polling() -> bool:
    """
    Whether to auto-enable force polling, it should be enabled automatically only on WSL.

    See samuelcolvin/watchfiles#187 for discussion.
    """
    import platform
    uname = platform.uname()
    return 'microsoft-standard' in uname.release.lower() and uname.system.lower() == 'linux'