from __future__ import annotations
from ._array_object import Array
from ._dtypes import _all_dtypes, _result_type
from dataclasses import dataclass
from typing import TYPE_CHECKING, List, Tuple, Union
import cupy as np
@dataclass
class finfo_object:
    bits: int
    eps: float
    max: float
    min: float
    smallest_normal: float