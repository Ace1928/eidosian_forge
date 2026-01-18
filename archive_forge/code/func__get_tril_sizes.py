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
def _get_tril_sizes(row: int, col: int, offset: int) -> Tuple[int, int, int]:
    if row == 0 or col == 0:
        return (0, 0, 0)
    m_first_row = min(col, 1 + offset) if offset > 0 else int(row + offset > 0)
    m_last_row = max(0, min(col, row + offset))
    n_row_all = max(0, min(row, row + offset))
    n_row_trapezoid = m_last_row - m_first_row + 1
    trapezoid_size = (m_first_row + m_last_row) * n_row_trapezoid // 2
    diff_row = n_row_all - n_row_trapezoid
    rectangle_size = max(0, diff_row * col)
    return (trapezoid_size, rectangle_size, m_first_row)