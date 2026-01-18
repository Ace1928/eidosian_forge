import contextlib
import dataclasses
import functools
import itertools
import logging
import operator
import re
from collections import namedtuple
from itertools import chain
from typing import (
import sympy
from sympy.printing.printer import Printer
import torch
import torch.fx
from torch.utils._sympy.value_ranges import ValueRanges
from .. import config, metrics
from ..utils import (
from ..virtualized import ops, OpsValue, V
@dataclasses.dataclass
class OptimizationContext:
    key: ClassVar[str] = 'opt_ctx'
    is_load_as_mask: bool = False
    dtype: Optional[torch.dtype] = None
    ops_name: str = ''
    is_most_inner_loop_irrevelant: bool = False
    is_load_uint8_as_float: bool = False