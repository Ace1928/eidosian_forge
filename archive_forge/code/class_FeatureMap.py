from abc import abstractmethod
from dataclasses import asdict, dataclass
from typing import Optional, Type, TypeVar
import torch
class FeatureMap(torch.nn.Module):

    def __init__(self, dim_features: int, iter_before_redraw: Optional[int]=None, normalize_inputs: bool=False, epsilon: float=1e-06):
        super().__init__()
        self.dim_features = dim_features
        self.dim_feature_map = dim_features
        self.iter_before_redraw = iter_before_redraw
        self.features: Optional[torch.Tensor] = None
        self.epsilon = epsilon
        self.normalize_inputs = normalize_inputs
        self._iter_counter = 0

    @abstractmethod
    def _get_feature_map(self, dim_input: int, dim_features: int, device: torch.device):
        raise NotImplementedError()

    @classmethod
    def from_config(cls: Type[Self], config: FeatureMapConfig) -> Self:
        fields = asdict(config)
        fields = {k: v for k, v in fields.items() if v is not None}
        return cls(**fields)