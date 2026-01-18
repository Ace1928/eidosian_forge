from __future__ import annotations
from collections import abc
from typing import TYPE_CHECKING, Any, Mapping, Optional, Sequence
from pymongo import max_staleness_selectors
from pymongo.errors import ConfigurationError
from pymongo.server_selectors import (
class MovingAverage:
    """Tracks an exponentially-weighted moving average."""
    average: Optional[float]

    def __init__(self) -> None:
        self.average = None

    def add_sample(self, sample: float) -> None:
        if sample < 0:
            return
        if self.average is None:
            self.average = sample
        else:
            self.average = 0.8 * self.average + 0.2 * sample

    def get(self) -> Optional[float]:
        """Get the calculated average, or None if no samples yet."""
        return self.average

    def reset(self) -> None:
        self.average = None