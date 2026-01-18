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
class _TrackThread(Thread):
    """A thread to periodically update progress."""

    def __init__(self, progress: 'Progress', task_id: 'TaskID', update_period: float):
        self.progress = progress
        self.task_id = task_id
        self.update_period = update_period
        self.done = Event()
        self.completed = 0
        super().__init__()

    def run(self) -> None:
        task_id = self.task_id
        advance = self.progress.advance
        update_period = self.update_period
        last_completed = 0
        wait = self.done.wait
        while not wait(update_period):
            completed = self.completed
            if last_completed != completed:
                advance(task_id, completed - last_completed)
                last_completed = completed
        self.progress.update(self.task_id, completed=self.completed, refresh=True)

    def __enter__(self) -> '_TrackThread':
        self.start()
        return self

    def __exit__(self, exc_type: Optional[Type[BaseException]], exc_val: Optional[BaseException], exc_tb: Optional[TracebackType]) -> None:
        self.done.set()
        self.join()