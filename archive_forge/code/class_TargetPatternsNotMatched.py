from __future__ import annotations
import collections
import collections.abc as c
import enum
import os
import re
import itertools
import abc
import typing as t
from .encoding import (
from .io import (
from .util import (
from .data import (
class TargetPatternsNotMatched(ApplicationError):
    """One or more targets were not matched when a match was required."""

    def __init__(self, patterns: set[str]) -> None:
        self.patterns = sorted(patterns)
        if len(patterns) > 1:
            message = 'Target patterns not matched:\n%s' % '\n'.join(self.patterns)
        else:
            message = 'Target pattern not matched: %s' % self.patterns[0]
        super().__init__(message)