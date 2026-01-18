from __future__ import annotations
import hashlib
import ntpath
import os
import os.path
import posixpath
import re
import sys
from typing import Callable, Iterable
from coverage import env
from coverage.exceptions import ConfigError
from coverage.misc import human_sorted, isolate_module, join_regex
class GlobMatcher:
    """A matcher for files by file name pattern."""

    def __init__(self, pats: Iterable[str], name: str='unknown') -> None:
        self.pats = list(pats)
        self.re = globs_to_regex(self.pats, case_insensitive=env.WINDOWS)
        self.name = name

    def __repr__(self) -> str:
        return f'<GlobMatcher {self.name} {self.pats!r}>'

    def info(self) -> list[str]:
        """A list of strings for displaying when dumping state."""
        return self.pats

    def match(self, fpath: str) -> bool:
        """Does `fpath` match one of our file name patterns?"""
        return self.re.match(fpath) is not None