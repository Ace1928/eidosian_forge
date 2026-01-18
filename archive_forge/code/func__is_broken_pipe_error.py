import contextlib
import errno
import logging
import logging.handlers
import os
import sys
import threading
from dataclasses import dataclass
from io import TextIOWrapper
from logging import Filter
from typing import Any, ClassVar, Generator, List, Optional, TextIO, Type
from pip._vendor.rich.console import (
from pip._vendor.rich.highlighter import NullHighlighter
from pip._vendor.rich.logging import RichHandler
from pip._vendor.rich.segment import Segment
from pip._vendor.rich.style import Style
from pip._internal.utils._log import VERBOSE, getLogger
from pip._internal.utils.compat import WINDOWS
from pip._internal.utils.deprecation import DEPRECATION_MSG_PREFIX
from pip._internal.utils.misc import ensure_dir
def _is_broken_pipe_error(exc_class: Type[BaseException], exc: BaseException) -> bool:
    if exc_class is BrokenPipeError:
        return True
    if not WINDOWS:
        return False
    return isinstance(exc, OSError) and exc.errno in (errno.EINVAL, errno.EPIPE)