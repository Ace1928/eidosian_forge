import io
import sys
import typing
import warnings
from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass, field
from datetime import timedelta
from io import RawIOBase, UnsupportedOperation
from math import ceil
from mmap import mmap
from operator import length_hint
from os import PathLike, stat
from threading import Event, RLock, Thread
from types import TracebackType
from typing import (
from . import filesize, get_console
from .console import Console, Group, JustifyMethod, RenderableType
from .highlighter import Highlighter
from .jupyter import JupyterMixin
from .live import Live
from .progress_bar import ProgressBar
from .spinner import Spinner
from .style import StyleType
from .table import Column, Table
from .text import Text, TextType
class _ReadContext(ContextManager[_I], Generic[_I]):
    """A utility class to handle a context for both a reader and a progress."""

    def __init__(self, progress: 'Progress', reader: _I) -> None:
        self.progress = progress
        self.reader: _I = reader

    def __enter__(self) -> _I:
        self.progress.start()
        return self.reader.__enter__()

    def __exit__(self, exc_type: Optional[Type[BaseException]], exc_val: Optional[BaseException], exc_tb: Optional[TracebackType]) -> None:
        self.progress.stop()
        self.reader.__exit__(exc_type, exc_val, exc_tb)