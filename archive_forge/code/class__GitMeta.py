from __future__ import annotations
import contextlib
import io
import itertools
import logging
import os
import re
import signal
import subprocess
from subprocess import DEVNULL, PIPE, Popen
import sys
from textwrap import dedent
import threading
import warnings
from git.compat import defenc, force_bytes, safe_decode
from git.exc import (
from git.util import (
from typing import (
from git.types import Literal, PathLike, TBD
class _GitMeta(type):
    """Metaclass for :class:`Git`.

    This helps issue :class:`DeprecationWarning` if :attr:`Git.USE_SHELL` is used.
    """

    def __getattribute(cls, name: str) -> Any:
        if name == 'USE_SHELL':
            _warn_use_shell(False)
        return super().__getattribute__(name)

    def __setattr(cls, name: str, value: Any) -> Any:
        if name == 'USE_SHELL':
            _warn_use_shell(value)
        super().__setattr__(name, value)
    if not TYPE_CHECKING:
        __getattribute__ = __getattribute
        __setattr__ = __setattr