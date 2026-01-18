from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple, Union
import torch
from torch._ops import OpOverload
from torch.distributed._tensor.placement_types import DTensorSpec
from torch.distributed.device_mesh import DeviceMesh
def _is_out_variant_op(op: OpOverload):
    return 'out' in op._schema.overload_name