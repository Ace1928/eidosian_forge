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
def _transform_interaction_constraints(self, value: Union[Sequence[Sequence[str]], str]) -> Union[str, List[List[int]]]:
    if isinstance(value, str):
        return value
    feature_idx_mapping = {name: idx for idx, name in enumerate(self.feature_names or [])}
    try:
        result = []
        for constraint in value:
            result.append([feature_idx_mapping[feature_name] for feature_name in constraint])
        return result
    except KeyError as e:
        raise ValueError('Constrained features are not a subset of training data feature names') from e