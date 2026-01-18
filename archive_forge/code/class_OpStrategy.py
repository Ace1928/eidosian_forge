from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple, Union
import torch
from torch._ops import OpOverload
from torch.distributed._tensor.placement_types import DTensorSpec
from torch.distributed.device_mesh import DeviceMesh
class OpStrategy(StrategyType):
    """
    OpStrategy that consists of a list of placement strategies associated with the op
    """

    def __init__(self, strategies: List[PlacementStrategy]) -> None:
        super().__init__()
        self.strategies: List[PlacementStrategy] = strategies

    def __str__(self) -> str:
        strategy_list_str = ', '.join([str(strategy) for strategy in self.strategies])
        mesh_shape = self.strategies[0].output_spec.mesh.shape
        return f'OpStrategy:[{strategy_list_str}] @mesh: {mesh_shape}'

    def max_num_shards(self) -> int:
        """
        Returns the max number of shards across all placement strategies
        """
        return max([strategy.output_spec.num_shards for strategy in self.strategies])

    @property
    def output_shape(self):
        return self.strategies[0].output_spec.shape

    @property
    def output_ndim(self):
        return self.strategies[0].output_spec.ndim