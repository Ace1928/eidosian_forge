from __future__ import annotations
import abc
import collections.abc as c
import enum
import fcntl
import importlib.util
import inspect
import json
import keyword
import os
import platform
import pkgutil
import random
import re
import shutil
import stat
import string
import subprocess
import sys
import time
import functools
import shlex
import typing as t
import warnings
from struct import unpack, pack
from termios import TIOCGWINSZ
from .locale_util import (
from .encoding import (
from .io import (
from .thread import (
from .constants import (
class OutputThread(ReaderThread):
    """Thread to pass stdout from a subprocess to stdout."""

    def _run(self) -> None:
        """Workload to run on a thread."""
        src = self.handle
        dst = self.buffer
        try:
            for line in src:
                dst.write(line)
                dst.flush()
        finally:
            src.close()