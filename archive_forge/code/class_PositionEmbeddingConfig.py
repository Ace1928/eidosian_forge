from abc import ABCMeta, abstractmethod
from dataclasses import asdict, dataclass
from typing import Type, TypeVar
import torch.nn as nn
@dataclass
class PositionEmbeddingConfig:
    name: str
    dim_model: int
    seq_len: int