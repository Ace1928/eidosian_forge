from typing import Any, List, Optional, Set, Tuple
import torch
from ..common import get_xformers_operator, register_operator
from .attn_bias import BlockDiagonalCausalWithOffsetPaddedKeysMask
from .common import AttentionFwOpBase, Context, Inputs

    An operator optimized for K=256 (so the contiguous dim fits into registers).
    Tested to work on MI250x.
    