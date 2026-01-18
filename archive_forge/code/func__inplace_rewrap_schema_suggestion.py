from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple, Union
import torch
from torch._ops import OpOverload
from torch.distributed._tensor.placement_types import DTensorSpec
from torch.distributed.device_mesh import DeviceMesh
def _inplace_rewrap_schema_suggestion(self, origin_schema: 'OpSchema') -> None:
    suggestion_args_spec = self.args_spec
    new_arg_schema: List[object] = []
    idx_of_args_spec = 0
    for arg in origin_schema.args_schema:
        if isinstance(arg, DTensorSpec):
            new_arg_schema.append(suggestion_args_spec[idx_of_args_spec])
            idx_of_args_spec += 1
        else:
            new_arg_schema.append(arg)
    self.args_schema = tuple(new_arg_schema)
    self.kwargs_schema = origin_schema.kwargs_schema