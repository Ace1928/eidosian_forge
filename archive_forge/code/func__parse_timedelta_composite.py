import datetime
import time
import re
import numbers
import functools
import contextlib
from numbers import Number
from typing import Union, Tuple, Iterable
from typing import cast
def _parse_timedelta_composite(raw_value, unit):
    if unit != 'seconds':
        raise ValueError('Cannot specify units with composite delta')
    values = raw_value.split(':')
    units = ('hours', 'minutes', 'seconds')
    composed = ' '.join((f'{value} {unit}' for value, unit in zip(values, units)))
    return _parse_timedelta_nanos(composed)