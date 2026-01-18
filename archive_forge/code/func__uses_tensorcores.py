from dataclasses import replace
from enum import Enum
from functools import partial
from typing import Any, List, Optional, Set, Tuple, Union
import torch
from ..common import get_xformers_operator, register_operator
from . import attn_bias
from .attn_bias import (
from .common import (
def _uses_tensorcores(sm: int, is_half: bool) -> bool:
    if sm >= 80:
        return True
    if sm >= 70:
        return is_half
    return False