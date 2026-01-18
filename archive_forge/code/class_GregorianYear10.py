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
class GregorianYear10(OrderedDateTime):
    """XSD 1.0 xs:gYear builtin type"""
    name = 'gYear'
    pattern = re.compile('^(?P<year>-?[0-9]*[0-9]{4})(?P<tzinfo>Z|[+-](?:(?:0[0-9]|1[0-3]):[0-5][0-9]|14:00))?$')

    def __init__(self, year: int, tzinfo: Optional[Timezone]=None) -> None:
        super(GregorianYear10, self).__init__(year, tzinfo=tzinfo)

    def __str__(self) -> str:
        return '{}{}'.format(self.iso_year, str(self.tzinfo or ''))