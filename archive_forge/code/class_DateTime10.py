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
class DateTime10(OrderedDateTime):
    """XSD 1.0 xs:dateTime builtin type"""
    name = 'dateTime'
    pattern = re.compile('^(?P<year>-?[0-9]*[0-9]{4})-(?P<month>[0-9]{2})-(?P<day>[0-9]{2})(T(?P<hour>[0-9]{2}):(?P<minute>[0-9]{2}):(?P<second>[0-9]{2})(?:\\.(?P<microsecond>[0-9]+))?)(?P<tzinfo>Z|[+-](?:(?:0[0-9]|1[0-3]):[0-5][0-9]|14:00))?$')

    def __init__(self, year: int, month: int, day: int, hour: int=0, minute: int=0, second: int=0, microsecond: int=0, tzinfo: Optional[datetime.tzinfo]=None) -> None:
        super(DateTime10, self).__init__(year, month, day, hour, minute, second, microsecond, tzinfo)

    def __str__(self) -> str:
        if self.microsecond:
            return '{}-{:02}-{:02}T{:02}:{:02}:{:02}.{}{}'.format(self.iso_year, self.month, self.day, self.hour, self.minute, self.second, '{:06}'.format(self.microsecond).rstrip('0'), str(self.tzinfo or ''))
        return '{}-{:02}-{:02}T{:02}:{:02}:{:02}{}'.format(self.iso_year, self.month, self.day, self.hour, self.minute, self.second, str(self.tzinfo or ''))