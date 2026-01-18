import argparse
import copy
import enum
import functools
import os
import typing
import warnings
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import timedelta
from typing import Any, Callable, Dict, Iterable, List, Literal, Optional, Tuple, get_args
import torch
from .constants import FSDP_AUTO_WRAP_POLICY, FSDP_BACKWARD_PREFETCH, FSDP_SHARDING_STRATEGY, FSDP_STATE_DICT_TYPE
from .environment import str_to_bool
from .imports import is_cuda_available, is_npu_available, is_xpu_available
from .versions import compare_versions
class PrecisionType(BaseEnum):
    """Represents a type of precision used on floating point values

    Values:

        - **NO** -- using full precision (FP32)
        - **FP16** -- using half precision
        - **BF16** -- using brain floating point precision
    """
    NO = 'no'
    FP8 = 'fp8'
    FP16 = 'fp16'
    BF16 = 'bf16'