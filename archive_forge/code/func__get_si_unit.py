from __future__ import annotations
import collections
import re
from functools import partial
from numbers import Number
from typing import TYPE_CHECKING, Any
import numpy as np
import scipy.constants as const
def _get_si_unit(unit):
    unit_type = _UNAME2UTYPE[unit]
    si_unit = filter(lambda k: BASE_UNITS[unit_type][k] == 1, BASE_UNITS[unit_type])
    return (next(iter(si_unit)), BASE_UNITS[unit_type][unit])