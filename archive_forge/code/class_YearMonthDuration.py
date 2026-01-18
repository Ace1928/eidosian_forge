from abc import abstractmethod
import math
import operator
import re
import datetime
from calendar import isleap
from decimal import Decimal, Context
from typing import cast, Any, Callable, Dict, Optional, Tuple, Union
from ..helpers import MONTH_DAYS_LEAP, MONTH_DAYS, DAYS_IN_4Y, \
from .atomic_types import AnyAtomicType
from .untyped import UntypedAtomic
class YearMonthDuration(Duration):
    name = 'yearMonthDuration'

    def __init__(self, months: int=0) -> None:
        super(YearMonthDuration, self).__init__(months, 0)

    def __repr__(self) -> str:
        return '%s(months=%r)' % (self.__class__.__name__, self.months)

    def __str__(self) -> str:
        m = abs(self.months)
        years, months = (m // 12, m % 12)
        if not years:
            return '-P%dM' % months if self.months < 0 else 'P%dM' % months
        elif not months:
            return '-P%dY' % years if self.months < 0 else 'P%dY' % years
        elif self.months < 0:
            return '-P%dY%dM' % (years, months)
        else:
            return 'P%dY%dM' % (years, months)

    def __add__(self, other: object) -> Union['YearMonthDuration', 'DayTimeDuration', 'OrderedDateTime']:
        if isinstance(other, self.__class__):
            return YearMonthDuration(months=self.months + other.months)
        elif isinstance(other, (DateTime10, Date10)):
            return other + self
        raise TypeError('cannot add %r to %r' % (type(other), type(self)))

    def __sub__(self, other: object) -> 'YearMonthDuration':
        if not isinstance(other, self.__class__):
            raise TypeError('cannot subtract %r from %r' % (type(other), type(self)))
        return YearMonthDuration(months=self.months - other.months)

    def __mul__(self, other: object) -> 'YearMonthDuration':
        if not isinstance(other, (float, int, Decimal)):
            raise TypeError('cannot multiply a %r by %r' % (type(self), type(other)))
        return YearMonthDuration(months=int(round_number(self.months * other)))

    def __truediv__(self, other: object) -> Union[float, 'YearMonthDuration']:
        if isinstance(other, self.__class__):
            return self.months / other.months
        elif isinstance(other, (float, int, Decimal)):
            return YearMonthDuration(months=int(round_number(self.months / other)))
        else:
            raise TypeError('cannot divide a %r by %r' % (type(self), type(other)))