import re
from contextlib import contextmanager
import functools
import operator
import warnings
import numbers
from collections import namedtuple
import inspect
import math
from typing import (
import numpy as np
from scipy._lib._array_api import array_namespace
def _get_nan(*data):
    data = [np.asarray(item) for item in data]
    try:
        dtype = np.result_type(*data, np.half)
    except DTypePromotionError:
        return np.array(np.nan, dtype=np.float64)[()]
    return np.array(np.nan, dtype=dtype)[()]