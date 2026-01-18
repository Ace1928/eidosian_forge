from abc import ABC, abstractmethod
from typing import Dict, Optional, Union
import torch
from .graph_manager import GraphManager
class MixingManager(ABC):

    def __init__(self, graph: GraphManager, device: Optional[torch.device]) -> None:
        self.graph_manager = graph
        self.device = device

    def is_regular(self) -> bool:
        """
        Whether there is bias accumulated in local entry of stationary
        distribution of mixing matrix
        """
        return self.graph_manager.is_regular_graph() and self.is_uniform()

    @abstractmethod
    def is_uniform(self) -> bool:
        """Whether mixing weights are distributed uniformly over peers"""
        raise NotImplementedError

    @abstractmethod
    def get_mixing_weights(self, residual_adjusted: bool=True) -> Dict[Union[str, int], torch.Tensor]:
        """Create mixing weight dictionary using uniform allocation"""
        raise NotImplementedError