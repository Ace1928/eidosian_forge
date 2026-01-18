import os.path
import re
from contextlib import suppress
from typing import TYPE_CHECKING, BinaryIO, Dict, Iterable, List, Optional, Union
from .config import Config, get_xdg_config_home_path
class IgnoreFilterStack:
    """Check for ignore status in multiple filters."""

    def __init__(self, filters) -> None:
        self._filters = filters

    def is_ignored(self, path: str) -> Optional[bool]:
        """Check whether a path is explicitly included or excluded in ignores.

        Args:
          path: Path to check
        Returns:
          None if the file is not mentioned, True if it is included,
          False if it is explicitly excluded.
        """
        status = None
        for filter in self._filters:
            status = filter.is_ignored(path)
            if status is not None:
                return status
        return status