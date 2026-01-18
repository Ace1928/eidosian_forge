from __future__ import annotations
from itertools import count
from .imports import symbol_by_name
class sorted_cycle(priority_cycle):
    """Cycle in sorted order."""

    def consume(self, n):
        """Consume n items."""
        return sorted(self.items[:n])