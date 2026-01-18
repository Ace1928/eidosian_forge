import builtins
import collections
import inspect
import itertools
import math
import operator
import warnings
from collections.abc import Iterable
from enum import Enum
from functools import partial, reduce, singledispatch, wraps
from typing import Any, Callable, Dict, List, Optional, overload, Sequence, Tuple, Union
import torch
import torch._prims as prims
import torch._prims_common as utils
from torch import sym_float, sym_int
from torch._prims_common import (
from torch._prims_common.wrappers import (
from torch._decomp import register_decomposition
import torch._refs._conversions
import torch._refs.fft
import torch._refs.linalg
import torch._refs.nn.functional
import torch._refs.special
def _get_triu_sizes(row: int, col: int, offset: int) -> Tuple[int, int, int]:
    if row == 0 or col == 0:
        return (0, 0, 0)
    m_first_row = max(0, col - offset) if offset > 0 else col
    rectangle_size = max(0, min(row, -offset) * col)
    trapezoid_size_tril, rectangle_size_tril, _ = _get_tril_sizes(row, col, offset - 1)
    triu_size = row * col - (trapezoid_size_tril + rectangle_size_tril)
    trapezoid_size = triu_size - rectangle_size
    return (trapezoid_size, rectangle_size, m_first_row)