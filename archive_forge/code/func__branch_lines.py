from __future__ import annotations
import collections
from typing import Callable, Iterable, TYPE_CHECKING
from coverage.debug import auto_repr
from coverage.exceptions import ConfigError
from coverage.misc import nice_pair
from coverage.types import TArc, TLineNo
def _branch_lines(self) -> list[TLineNo]:
    """Returns a list of line numbers that have more than one exit."""
    return [l1 for l1, count in self.exit_counts.items() if count > 1]