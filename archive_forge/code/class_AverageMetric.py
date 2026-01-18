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
class AverageMetric(Metric):
    """
    Class that keeps a running average of some metric.

    Examples of AverageMetrics include hits@1, F1, accuracy, etc. These metrics all have
    per-example values that can be directly mapped back to a teacher.
    """
    __slots__ = ('_numer', '_denom')

    @property
    def macro_average(self) -> bool:
        """
        Indicates whether this metric should be macro-averaged when globally reported.
        """
        return True

    def __init__(self, numer: TScalar, denom: TScalar=1):
        self._numer = self.as_number(numer)
        self._denom = self.as_number(denom)

    def __add__(self, other: Optional['AverageMetric']) -> 'AverageMetric':
        if other is None:
            return self
        full_numer: TScalar = self._numer + other._numer
        full_denom: TScalar = self._denom + other._denom
        return type(self)(numer=full_numer, denom=full_denom)

    def value(self) -> float:
        if self._numer == 0 and self._denom == 0:
            return 0.0
        if self._denom == 0:
            return float('nan')
        return self._numer / self._denom