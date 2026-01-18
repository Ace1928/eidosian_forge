import os
import os.path
import pathlib
import posixpath
import stat
import sys
import warnings
from collections.abc import (
from dataclasses import (
from os import (
from typing import (
from .pattern import (
def _filter_check_patterns(patterns: Iterable[Pattern]) -> List[Tuple[int, Pattern]]:
    """
	Filters out null-patterns.

	*patterns* (:class:`Iterable` of :class:`.Pattern`) contains the
	patterns.

	Returns a :class:`list` containing each indexed pattern (:class:`tuple`) which
	contains the pattern index (:class:`int`) and the actual pattern
	(:class:`~pathspec.pattern.Pattern`).
	"""
    return [(__index, __pat) for __index, __pat in enumerate(patterns) if __pat.include is not None]