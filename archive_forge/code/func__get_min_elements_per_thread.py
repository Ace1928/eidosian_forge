from __future__ import annotations
import collections
import contextlib
import dataclasses
import functools
import itertools
import logging
import math
import operator
import os
import textwrap
from typing import Any, Counter, Dict, Iterable, List, Optional, Set, Tuple, Union
import sympy
import torch
import torch._logging
from torch._prims_common import is_integer_dtype
from torch.utils._sympy.functions import FloorDiv, ModularIndexing
from torch.utils._sympy.value_ranges import ValueRanges
from ..._dynamo.utils import counters
from .. import config, ir, scheduler
from ..codecache import code_hash, get_path, PyCodeCache
from ..dependencies import MemoryDep, StarDep
from ..ir import IRNode, ReductionHint, TritonTemplateBuffer
from ..optimize_indexing import indexing_dtype_strength_reduction
from ..scheduler import BaseScheduling, WhyNoFuse
from ..triton_heuristics import AutotuneHint
from ..utils import (
from ..virtualized import ops, V
from ..wrapper_benchmark import get_kernel_category_by_source_code
from .common import (
from .triton_utils import config_of, signature_of, signature_to_meta
def _get_min_elements_per_thread(src_dtype: torch.dtype, dst_dtype: torch.dtype) -> int:
    if src_dtype == dst_dtype:
        return 0
    fp8_dtypes = {torch.float8_e4m3fn, torch.float8_e5m2}
    assert not (src_dtype in fp8_dtypes and dst_dtype in fp8_dtypes and (src_dtype != dst_dtype)), 'Conversions between float8_e5m2 and float8_e4m3fn is not supported!'
    if src_dtype == torch.float8_e5m2 or dst_dtype == torch.float8_e5m2:
        return 4
    if src_dtype == torch.float8_e4m3fn or dst_dtype == torch.float8_e4m3fn:
        return 2
    return 0