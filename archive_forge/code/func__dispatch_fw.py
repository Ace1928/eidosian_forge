import textwrap
from collections import deque
from typing import List, Sequence, Type, TypeVar
import torch
from . import (
from .common import AttentionBwOpBase, AttentionFwOpBase, Inputs
def _dispatch_fw(inp: Inputs, needs_gradient: bool) -> Type[AttentionFwOpBase]:
    """Computes the best operator for forward

    Raises:
        NotImplementedError: if not operator was found

    Returns:
        AttentionOp: The best operator for the configuration
    """
    return _run_priority_list('memory_efficient_attention_forward', _dispatch_fw_priority_list(inp, needs_gradient), inp)