from __future__ import annotations
from pydantic import BaseModel, Field
from typing import List, Optional, Union, Tuple, Dict, Iterator, Any, Type, Set, Iterable, TYPE_CHECKING
from .debug import get_autologger
class MonetaryMetric(NumericValuesContainer):
    """
    A container for monetary values
    """
    name: Optional[str] = 'costs'
    unit: Optional[str] = 'USD'
    symbol: Optional[str] = '$'

    def pretty(self, value: float) -> str:
        """
        Returns the pretty representation of the value
        """
        return f'{self.symbol}{value:.2f}'

    @property
    def total_s(self) -> str:
        """
        Returns the total in s
        """
        return self.pretty(self.total)

    @property
    def average_s(self) -> str:
        """
        Returns the average in s
        """
        return self.pretty(self.average)

    @property
    def median_s(self) -> str:
        """
        Returns the median in s
        """
        return self.pretty(self.median)

    def __repr__(self) -> str:
        """
        Returns the representation of the values
        """
        d = {self.name: {'total': self.total_s, 'average': self.average_s, 'median': self.median_s, 'count': self.count}}
        return str(d)

    def __str__(self) -> str:
        """
        Returns the string representation of the values
        """
        return f'{self.total_s}, {self.average_s}, {self.count}'