import abc
import code
import inspect
import os
import pkgutil
import pydoc
import shlex
import subprocess
import sys
import tempfile
import textwrap
import time
import traceback
from abc import abstractmethod
from dataclasses import dataclass
from itertools import takewhile
from pathlib import Path
from types import ModuleType, TracebackType
from typing import (
from ._typing_compat import Literal
from pygments.lexers import Python3Lexer
from pygments.token import Token, _TokenType
from . import autocomplete, inspection, simpleeval
from .config import getpreferredencoding, Config
from .formatter import Parenthesis
from .history import History
from .lazyre import LazyReCompile
from .paste import PasteHelper, PastePinnwand, PasteFailed
from .patch_linecache import filename_for_console_input
from .translations import _, ngettext
from .importcompletion import ModuleGatherer
class RuntimeTimer:
    """Calculate running time"""

    def __init__(self) -> None:
        self.reset_timer()

    def __enter__(self) -> None:
        self.start = time.monotonic()

    def __exit__(self, exc_type: Optional[Type[BaseException]], exc_val: Optional[BaseException], exc_tb: Optional[TracebackType]) -> Literal[False]:
        self.last_command = time.monotonic() - self.start
        self.running_time += self.last_command
        return False

    def reset_timer(self) -> None:
        self.running_time = 0.0
        self.last_command = 0.0

    def estimate(self) -> float:
        return self.running_time - self.last_command