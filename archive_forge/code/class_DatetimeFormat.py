import functools
import numbers
import sys
import numpy as np
from . import numerictypes as _nt
from .umath import absolute, isinf, isfinite, isnat
from . import multiarray
from .multiarray import (array, dragon4_positional, dragon4_scientific,
from .fromnumeric import any
from .numeric import concatenate, asarray, errstate
from .numerictypes import (longlong, intc, int_, float_, complex_, bool_,
from .overrides import array_function_dispatch, set_module
import operator
import warnings
import contextlib
class DatetimeFormat(_TimelikeFormat):

    def __init__(self, x, unit=None, timezone=None, casting='same_kind', legacy=False):
        if unit is None:
            if x.dtype.kind == 'M':
                unit = datetime_data(x.dtype)[0]
            else:
                unit = 's'
        if timezone is None:
            timezone = 'naive'
        self.timezone = timezone
        self.unit = unit
        self.casting = casting
        self.legacy = legacy
        super().__init__(x)

    def __call__(self, x):
        if self.legacy <= 113:
            return self._format_non_nat(x)
        return super().__call__(x)

    def _format_non_nat(self, x):
        return "'%s'" % datetime_as_string(x, unit=self.unit, timezone=self.timezone, casting=self.casting)