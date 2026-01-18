from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple, Union
import torch
from torch._ops import OpOverload
from torch.distributed._tensor.placement_types import DTensorSpec
from torch.distributed.device_mesh import DeviceMesh
@dataclass
class OpInfo:
    """
    All Runtime Op execution info are packed here
    """
    mesh: DeviceMesh
    schema: OpSchema
    flat_args_schema: List[object]
    local_args: Sequence[object]
    local_kwargs: Dict[str, object]
    args_tree_spec: Optional[TreeSpec] = None
    output_sharding: Optional[OutputSharding] = None