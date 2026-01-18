from abc import ABC, abstractmethod
from typing import Dict, Optional, Union
import torch
from .graph_manager import GraphManager
class UniformMixing(MixingManager):

    def get_mixing_weights(self, residual_adjusted: bool=True) -> Dict[Union[str, int], torch.Tensor]:
        """Create mixing weight dictionary using uniform allocation"""
        mixing_weights: Dict[Union[str, int], torch.Tensor] = {}
        out_peers, _ = self.graph_manager.get_peers()
        w = torch.tensor([1.0 / (len(out_peers) + 1.0)], device=self.device)
        mixing_weights['lo'] = w.clone()
        w_op = w if not residual_adjusted else w / mixing_weights['lo']
        mixing_weights['uniform'] = w_op.clone()
        for op in out_peers:
            mixing_weights[op] = w_op.clone()
        return mixing_weights

    def is_uniform(self) -> bool:
        return True