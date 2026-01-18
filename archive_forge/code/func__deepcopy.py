import copy
import os
from functools import partial
from itertools import groupby
from typing import TYPE_CHECKING, Callable, Iterator, List, Optional, Tuple, TypeVar, Union
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.types
from . import config
from .utils.logging import get_logger
def _deepcopy(x, memo: dict):
    """deepcopy a regular class instance"""
    cls = x.__class__
    result = cls.__new__(cls)
    memo[id(x)] = result
    for k, v in x.__dict__.items():
        setattr(result, k, copy.deepcopy(v, memo))
    return result