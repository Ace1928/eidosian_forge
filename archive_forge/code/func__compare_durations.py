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
def _compare_durations(self, other: object, op: Callable[[Any, Any], Any]) -> bool:
    """
        Ordering is defined through comparison of four datetime.datetime values.

        Ref: https://www.w3.org/TR/2012/REC-xmlschema11-2-20120405/#duration
        """
    if not isinstance(other, self.__class__):
        raise TypeError('wrong type %r for operand %r' % (type(other), other))
    m1, s1 = (self.months, int(self.seconds))
    m2, s2 = (other.months, int(other.seconds))
    ms1, ms2 = (int((self.seconds - s1) * 1000000), int((other.seconds - s2) * 1000000))
    return all([op(datetime.timedelta(months2days(1696, 9, m1), s1, ms1), datetime.timedelta(months2days(1696, 9, m2), s2, ms2)), op(datetime.timedelta(months2days(1697, 2, m1), s1, ms1), datetime.timedelta(months2days(1697, 2, m2), s2, ms2)), op(datetime.timedelta(months2days(1903, 3, m1), s1, ms1), datetime.timedelta(months2days(1903, 3, m2), s2, ms2)), op(datetime.timedelta(months2days(1903, 7, m1), s1, ms1), datetime.timedelta(months2days(1903, 7, m2), s2, ms2))])