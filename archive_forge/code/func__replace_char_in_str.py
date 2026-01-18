from typing import cast, Dict, List, Optional, Tuple
import torch
from torch.distributed._tensor._utils import compute_local_shape
from torch.distributed._tensor.op_schema import (
from torch.distributed._tensor.ops.utils import prod
from torch.distributed._tensor.placement_types import DTensorSpec, TensorMeta
def _replace_char_in_str(string: str, new_char: str, idx: int) -> str:
    return string[:idx] + new_char + string[idx + 1:]