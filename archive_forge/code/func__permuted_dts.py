import pytest
from contextlib import nullcontext
import datetime
import itertools
import pytz
from traitlets import TraitError
from ..widget_datetime import DatetimePicker
def _permuted_dts():
    ret = []
    combos = list(itertools.product([None, dt_1442, dt_2002, dt_2056], repeat=3))
    for vals in combos:
        expected = vals[0]
        if vals[1] and vals[2] and (vals[1] > vals[2]):
            expected = TraitError
        elif vals[0] is None:
            pass
        elif vals[1] and vals[1] > vals[0]:
            expected = vals[1]
        elif vals[2] and vals[2] < vals[0]:
            expected = vals[2]
        ret.append(vals + (expected,))
    return ret