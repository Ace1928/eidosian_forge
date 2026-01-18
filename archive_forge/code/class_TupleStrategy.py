from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple, Union
import torch
from torch._ops import OpOverload
from torch.distributed._tensor.placement_types import DTensorSpec
from torch.distributed.device_mesh import DeviceMesh
class TupleStrategy(StrategyType):
    """
    TupleStrategy represents the output strategy of this op is a tuple
    of strategy, i.e. If the output of this op is a tuple of tensors or list of tensors
    with possibly different placement strategies, we should return a TupleStrategy that
    contains a tuple of OpStrategy.

    NOTE: if the output of the op is a List[Tensor] and they share the same placement
    strategy, then we should return a single OpStrategy instead of a TupleStrategy
    """

    def __init__(self, childs: Sequence[StrategyType]) -> None:
        super().__init__()
        self.childs: Sequence[StrategyType] = childs

    def __str__(self) -> str:
        child_strategies_str = ', '.join([f'{str(strat)}' for idx, strat in enumerate(self.childs)])
        return f'TupleStrategy({child_strategies_str})'