import logging
import os
import re
from pathlib import Path
from typing import TYPE_CHECKING, Optional, Sequence, Union
class BaseFilter:
    """
    Useful base class for creating filters. `BaseFilter` should be inherited and configured, rather than used
    directly.

    The class supports ignoring files in 3 ways:
    """
    __slots__ = ('_ignore_dirs', '_ignore_entity_regexes', '_ignore_paths')
    ignore_dirs: Sequence[str] = ()
    'Full names of directories to ignore, an obvious example would be `.git`.'
    ignore_entity_patterns: Sequence[str] = ()
    '\n    Patterns of files or directories to ignore, these are compiled into regexes.\n\n    "entity" here refers to the specific file or directory - basically the result of `path.split(os.sep)[-1]`,\n    an obvious example would be `r\'\\.py[cod]$\'`.\n    '
    ignore_paths: Sequence[Union[str, Path]] = ()
    '\n    Full paths to ignore, e.g. `/home/users/.cache` or `C:\\Users\\user\\.cache`.\n    '

    def __init__(self) -> None:
        self._ignore_dirs = set(self.ignore_dirs)
        self._ignore_entity_regexes = tuple((re.compile(r) for r in self.ignore_entity_patterns))
        self._ignore_paths = tuple(map(str, self.ignore_paths))

    def __call__(self, change: 'Change', path: str) -> bool:
        """
        Instances of `BaseFilter` subclasses can be used as callables.
        Args:
            change: The type of change that occurred, see [`Change`][watchfiles.Change].
            path: the raw path of the file or directory that changed.

        Returns:
            True if the file should be included in changes, False if it should be ignored.
        """
        parts = path.lstrip(os.sep).split(os.sep)
        if any((p in self._ignore_dirs for p in parts)):
            return False
        entity_name = parts[-1]
        if any((r.search(entity_name) for r in self._ignore_entity_regexes)):
            return False
        elif self._ignore_paths and path.startswith(self._ignore_paths):
            return False
        else:
            return True

    def __repr__(self) -> str:
        args = ', '.join((f'{k}={getattr(self, k, None)!r}' for k in self.__slots__))
        return f'{self.__class__.__name__}({args})'