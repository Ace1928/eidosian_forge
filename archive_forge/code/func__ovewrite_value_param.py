import functools
import inspect
import warnings
from collections import OrderedDict
from typing import Any, Callable, Dict, Optional, Tuple, TypeVar, Union
from torch import nn
from .._utils import sequence_to_str
from ._api import WeightsEnum
def _ovewrite_value_param(param: str, actual: Optional[V], expected: V) -> V:
    if actual is not None:
        if actual != expected:
            raise ValueError(f"The parameter '{param}' expected value {expected} but got {actual} instead.")
    return expected