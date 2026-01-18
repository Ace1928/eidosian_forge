import datetime
import time
import re
import numbers
import functools
import contextlib
from numbers import Number
from typing import Union, Tuple, Iterable
from typing import cast
class _Saved_NS:
    """
    Bundle a timedelta with nanoseconds.

    >>> _Saved_NS.derive('microseconds', .001)
    _Saved_NS(td=datetime.timedelta(0), nanoseconds=1)
    """
    td = datetime.timedelta()
    nanoseconds = 0
    multiplier = dict(seconds=1000000000, milliseconds=1000000, microseconds=1000)

    def __init__(self, **kwargs):
        vars(self).update(kwargs)

    @classmethod
    def derive(cls, unit, value):
        if unit == 'nanoseconds':
            return _Saved_NS(nanoseconds=value)
        try:
            raw_td = datetime.timedelta(**{unit: value})
        except TypeError:
            raise ValueError(f'Invalid unit {unit}')
        res = _Saved_NS(td=raw_td)
        with contextlib.suppress(KeyError):
            res.nanoseconds = int(value * cls.multiplier[unit]) % 1000
        return res

    def __add__(self, other):
        return _Saved_NS(td=self.td + other.td, nanoseconds=self.nanoseconds + other.nanoseconds)

    def resolve(self):
        """
        Resolve any nanoseconds into the microseconds field,
        discarding any nanosecond resolution (but honoring partial
        microseconds).
        """
        addl_micros = round(self.nanoseconds / 1000)
        return self.td + datetime.timedelta(microseconds=addl_micros)

    def __repr__(self):
        return f'_Saved_NS(td={self.td!r}, nanoseconds={self.nanoseconds!r})'