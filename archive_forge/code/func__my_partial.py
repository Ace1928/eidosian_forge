from __future__ import annotations
import collections
import re
from functools import partial
from numbers import Number
from typing import TYPE_CHECKING, Any
import numpy as np
import scipy.constants as const
def _my_partial(func, *args, **kwargs):
    """Partial returns a partial object and therefore we cannot inherit class
    methods defined in FloatWithUnit. This function calls partial and patches
    the new class before returning.
    """
    newobj = partial(func, *args, **kwargs)
    newobj.from_str = FloatWithUnit.from_str
    return newobj