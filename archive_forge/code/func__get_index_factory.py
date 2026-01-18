from __future__ import annotations
from contextlib import suppress
import copy
from datetime import (
import itertools
import os
import re
from textwrap import dedent
from typing import (
import warnings
import numpy as np
from pandas._config import (
from pandas._libs import (
from pandas._libs.lib import is_string_array
from pandas._libs.tslibs import timezones
from pandas.compat._optional import import_optional_dependency
from pandas.compat.pickle_compat import patch_pickle
from pandas.errors import (
from pandas.util._decorators import cache_readonly
from pandas.util._exceptions import find_stack_level
from pandas.core.dtypes.common import (
from pandas.core.dtypes.dtypes import (
from pandas.core.dtypes.missing import array_equivalent
from pandas import (
from pandas.core.arrays import (
import pandas.core.common as com
from pandas.core.computation.pytables import (
from pandas.core.construction import extract_array
from pandas.core.indexes.api import ensure_index
from pandas.core.internals import (
from pandas.io.common import stringify_path
from pandas.io.formats.printing import (
def _get_index_factory(self, attrs):
    index_class = self._alias_to_class(_ensure_decoded(getattr(attrs, 'index_class', '')))
    factory: Callable
    if index_class == DatetimeIndex:

        def f(values, freq=None, tz=None):
            dta = DatetimeArray._simple_new(values.values, dtype=values.dtype, freq=freq)
            result = DatetimeIndex._simple_new(dta, name=None)
            if tz is not None:
                result = result.tz_localize('UTC').tz_convert(tz)
            return result
        factory = f
    elif index_class == PeriodIndex:

        def f(values, freq=None, tz=None):
            dtype = PeriodDtype(freq)
            parr = PeriodArray._simple_new(values, dtype=dtype)
            return PeriodIndex._simple_new(parr, name=None)
        factory = f
    else:
        factory = index_class
    kwargs = {}
    if 'freq' in attrs:
        kwargs['freq'] = attrs['freq']
        if index_class is Index:
            factory = TimedeltaIndex
    if 'tz' in attrs:
        if isinstance(attrs['tz'], bytes):
            kwargs['tz'] = attrs['tz'].decode('utf-8')
        else:
            kwargs['tz'] = attrs['tz']
        assert index_class is DatetimeIndex
    return (factory, kwargs)