from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple, Union
import torch
from torch._ops import OpOverload
from torch.distributed._tensor.placement_types import DTensorSpec
from torch.distributed.device_mesh import DeviceMesh
@dataclass
class PlacementStrategy:
    """
    A placement strategy describes an acceptable sharding placements of the output
    and the tensor arguments of an operation.
    """
    output_spec: DTensorSpec
    input_specs: Optional[Sequence[DTensorSpec]] = None
    redistribute_cost: Optional[List[List[float]]] = None

    def pretty_print_placements(self, placements):
        return ''.join([str(p) for p in placements])

    def __str__(self) -> str:
        if self.input_specs is None:
            input_specs_str = ''
        else:
            input_specs_str = '(' + ', '.join([self.pretty_print_placements(spec.placements) for spec in self.input_specs]) + ') -> '
        output_spec_str = self.pretty_print_placements(self.output_spec.placements)
        return f'{input_specs_str}{output_spec_str}'