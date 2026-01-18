import errno
import itertools
import logging
import os.path
import tempfile
import traceback
from contextlib import ExitStack, contextmanager
from pathlib import Path
from typing import (
from pip._internal.utils.misc import enum, rmtree
class TempDirectoryTypeRegistry:
    """Manages temp directory behavior"""

    def __init__(self) -> None:
        self._should_delete: Dict[str, bool] = {}

    def set_delete(self, kind: str, value: bool) -> None:
        """Indicate whether a TempDirectory of the given kind should be
        auto-deleted.
        """
        self._should_delete[kind] = value

    def get_delete(self, kind: str) -> bool:
        """Get configured auto-delete flag for a given TempDirectory type,
        default True.
        """
        return self._should_delete.get(kind, True)