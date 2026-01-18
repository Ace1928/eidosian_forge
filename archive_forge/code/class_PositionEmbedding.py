from abc import ABCMeta, abstractmethod
from dataclasses import asdict, dataclass
from typing import Type, TypeVar
import torch.nn as nn
class PositionEmbedding(nn.Module, metaclass=ABCMeta):

    @abstractmethod
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()

    @classmethod
    def from_config(cls: Type[Self], config: PositionEmbeddingConfig) -> Self:
        fields = asdict(config)
        fields = {k: v for k, v in fields.items() if v is not None}
        return cls(**fields)