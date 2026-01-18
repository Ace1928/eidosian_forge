from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable
import numpy as np
import scipy.sparse as sp
from scipy.signal import convolve
from cvxpy.lin_ops import LinOp
from cvxpy.settings import (
def add_dicts(self, a: dict, b: dict) -> dict:
    """
        Addition for dict-based tensors.
        """
    res = {}
    keys_a = set(a.keys())
    keys_b = set(b.keys())
    intersect = keys_a & keys_b
    for key in intersect:
        if isinstance(a[key], dict) and isinstance(b[key], dict):
            res[key] = self.add_dicts(a[key], b[key])
        elif isinstance(a[key], self.tensor_type()) and isinstance(b[key], self.tensor_type()):
            res[key] = self.add_tensors(a[key], b[key])
        else:
            raise ValueError(f'Values must either be dicts or {self.tensor_type()}.')
    for key in keys_a - intersect:
        res[key] = a[key]
    for key in keys_b - intersect:
        res[key] = b[key]
    return res