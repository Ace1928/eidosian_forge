from typing import Any, Dict, List, Optional, Generic, TypeVar, cast
from types import TracebackType
from importlib.metadata import entry_points
from toolz import curry
class NoSuchEntryPoint(Exception):

    def __init__(self, group, name):
        self.group = group
        self.name = name

    def __str__(self):
        return f'No {self.name!r} entry point found in group {self.group!r}'