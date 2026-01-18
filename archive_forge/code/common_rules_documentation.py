from typing import cast, Dict, List, Optional, Tuple
import torch
from torch.distributed._tensor._utils import compute_local_shape
from torch.distributed._tensor.op_schema import (
from torch.distributed._tensor.ops.utils import prod
from torch.distributed._tensor.placement_types import DTensorSpec, TensorMeta

    Propagate the sharding for pointwise operations.

    Examples:
        ij,ij->ij - addition/mul
        ij,j->ij - broadcasted addition
    