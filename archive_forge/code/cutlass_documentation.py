from dataclasses import replace
from enum import Enum
from functools import partial
from typing import Any, List, Optional, Set, Tuple, Union
import torch
from ..common import get_xformers_operator, register_operator
from . import attn_bias
from .attn_bias import (
from .common import (
xFormers' MHA kernel based on CUTLASS.
    Supports a large number of settings (including without TensorCores, f32 ...)
    and GPUs as old as P100 (Sm60)
    