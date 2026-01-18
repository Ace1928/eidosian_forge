import copy
import ctypes
import importlib.util
import json
import os
import re
import sys
import warnings
import weakref
from abc import ABC, abstractmethod
from collections.abc import Mapping
from enum import IntEnum, unique
from functools import wraps
from inspect import Parameter, signature
from typing import (
import numpy as np
import scipy.sparse
from ._typing import (
from .compat import PANDAS_INSTALLED, DataFrame, py_str
from .libpath import find_lib_path
def _check_distributed_params(kwargs: Dict[str, Any]) -> None:
    """Validate parameters in distributed environments."""
    device = kwargs.get('device', None)
    if device and (not isinstance(device, str)):
        msg = 'Invalid type for the `device` parameter'
        msg += _expect((str,), type(device))
        raise TypeError(msg)
    if device and device.find(':') != -1:
        raise ValueError("Distributed training doesn't support selecting device ordinal as GPUs are managed by the distributed framework. use `device=cuda` or `device=gpu` instead.")
    if kwargs.get('booster', None) == 'gblinear':
        raise NotImplementedError(f'booster `{kwargs['booster']}` is not supported for distributed training.')