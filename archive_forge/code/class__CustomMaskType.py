from dataclasses import replace
from enum import Enum
from functools import partial
from typing import Any, List, Mapping, Optional, Set, Tuple, Union
import torch
from ..common import get_xformers_operator, register_operator
from . import attn_bias
from .attn_bias import (
from .common import (
class _CustomMaskType(int, Enum):
    """
    (Matches CustomMaskType in C++.)
    """
    NoCustomMask = 0
    CausalFromTopLeft = 1
    CausalFromBottomRight = 2