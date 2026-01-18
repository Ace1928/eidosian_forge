from __future__ import annotations
import functools
import time
from collections import deque
from contextlib import AbstractContextManager
from contextvars import ContextVar, Token
from typing import Any, Callable, Deque, MutableMapping, Optional, TypeVar, cast
from pymongo.write_concern import WriteConcern
class MovingMinimum:
    """Tracks a minimum RTT within the last 10 RTT samples."""
    samples: Deque[float]

    def __init__(self) -> None:
        self.samples = deque(maxlen=_MAX_RTT_SAMPLES)

    def add_sample(self, sample: float) -> None:
        if sample < 0:
            return
        self.samples.append(sample)

    def get(self) -> float:
        """Get the min, or 0.0 if there aren't enough samples yet."""
        if len(self.samples) >= _MIN_RTT_SAMPLES:
            return min(self.samples)
        return 0.0

    def reset(self) -> None:
        self.samples.clear()