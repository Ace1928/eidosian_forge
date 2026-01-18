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
def _prep_changes(raw_changes: Set[Tuple[int, str]], watch_filter: Optional[Callable[[Change, str], bool]]) -> Set[FileChange]:
    changes = {(Change(change), path) for change, path in raw_changes}
    if watch_filter:
        changes = {c for c in changes if watch_filter(c[0], c[1])}
    return changes