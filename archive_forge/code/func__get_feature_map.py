from abc import abstractmethod
from dataclasses import asdict, dataclass
from typing import Optional, Type, TypeVar
import torch
@abstractmethod
def _get_feature_map(self, dim_input: int, dim_features: int, device: torch.device):
    raise NotImplementedError()