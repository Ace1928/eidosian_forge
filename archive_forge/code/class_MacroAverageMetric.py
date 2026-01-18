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
class MacroAverageMetric(Metric):
    """
    Class that represents the macro average of several numbers.

    Used for aggregating task level metrics. It is only used for things that are
    AverageMetrics already.
    """
    __slots__ = '_values'

    def __init__(self, metrics: Dict[str, Metric]) -> None:
        self._values = metrics

    def __add__(self, other: Optional['MacroAverageMetric']) -> 'MacroAverageMetric':
        if other is None:
            return self
        output = dict(**self._values)
        for k, v in other._values.items():
            output[k] = output.get(k, None) + v
        return MacroAverageMetric(output)

    def value(self) -> float:
        sum_ = sum((v.value() for v in self._values.values()))
        n = len(self._values)
        return sum_ / n