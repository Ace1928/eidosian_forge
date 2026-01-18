import textwrap
from collections import deque
from typing import List, Sequence, Type, TypeVar
import torch
from . import (
from .common import AttentionBwOpBase, AttentionFwOpBase, Inputs
def _format_not_supported_reasons(op, reasons: List[str]) -> str:
    return f'`{op.NAME}` is not supported because:\n    ' + '\n    '.join(reasons)