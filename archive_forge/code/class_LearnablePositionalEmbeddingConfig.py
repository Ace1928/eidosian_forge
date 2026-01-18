from dataclasses import dataclass
import torch
from xformers.components.positional_embedding import (
@dataclass
class LearnablePositionalEmbeddingConfig(PositionEmbeddingConfig):
    name: str
    seq_len: int
    dim_model: int
    add_class_token: bool