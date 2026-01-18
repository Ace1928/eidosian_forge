import datetime
import functools
import logging
import re
from dateutil.rrule import (rrule, MO, TU, WE, TH, FR, SA, SU, YEARLY,
from dateutil.relativedelta import relativedelta
import dateutil.parser
import dateutil.tz
import numpy as np
import matplotlib as mpl
from matplotlib import _api, cbook, ticker, units
def _aware_return_wrapper(self, f, returns_list=False):
    """Decorator function that allows rrule methods to handle tzinfo."""
    if self._tzinfo is None:
        return f

    def normalize_arg(arg):
        if isinstance(arg, datetime.datetime) and arg.tzinfo is not None:
            if arg.tzinfo is not self._tzinfo:
                arg = arg.astimezone(self._tzinfo)
            return arg.replace(tzinfo=None)
        return arg

    def normalize_args(args, kwargs):
        args = tuple((normalize_arg(arg) for arg in args))
        kwargs = {kw: normalize_arg(arg) for kw, arg in kwargs.items()}
        return (args, kwargs)
    if not returns_list:

        def inner_func(*args, **kwargs):
            args, kwargs = normalize_args(args, kwargs)
            dt = f(*args, **kwargs)
            return self._attach_tzinfo(dt, self._tzinfo)
    else:

        def inner_func(*args, **kwargs):
            args, kwargs = normalize_args(args, kwargs)
            dts = f(*args, **kwargs)
            return [self._attach_tzinfo(dt, self._tzinfo) for dt in dts]
    return functools.wraps(f)(inner_func)