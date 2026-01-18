import re
from abc import ABC, abstractmethod
from collections import Counter
import functools
import datetime
from typing import Union, List, Optional, Tuple, Set, Any, Dict
import torch
from parlai.core.message import Message
from parlai.utils.misc import warn_once
from parlai.utils.typing import TScalar, TVector
class TimerMetric(Metric):
    """
    A timer metric keep tracks of the first/last times it was used.
    """
    __slots__ = ('_value', '_start', '_end')

    @classmethod
    def _now(cls) -> int:
        return datetime.datetime.utcnow().timestamp()

    def __init__(self, value: TScalar, start_time: Optional[int]=None, end_time: Optional[int]=None):
        self._value = self.as_number(value)
        if start_time is None:
            start_time = self._now()
        if end_time is None:
            end_time = self._now()
        self._start = start_time
        self._end = end_time

    def __add__(self, other: Optional['TimerMetric']) -> 'TimerMetric':
        if other is None:
            return self
        total: TScalar = self._value + other._value
        start: int = min(self._start, other._start)
        end: int = max(self._start, other._end)
        return type(self)(total, start, end)

    def value(self) -> float:
        if self._value == 0 or self._end == self._start:
            return 0
        return self._value / (self._end - self._start)