import math
from dataclasses import dataclass
from enum import Enum
import torch
@dataclass
class PatchEmbeddingConfig:
    """
    The configuration for the patch embedding layer, which takes the raw token passed in
    and returns a pooled representation along a given embedding dimension.

    This typically trades the spatial (context length) representation with the embedding size

    This is canonicaly used by ViT, but other papers (like MetaFormer or other hierarchical transformers)
    propose a more general use case for this
    """
    in_channels: int
    out_channels: int
    kernel_size: int
    stride: int
    padding: int = 0
    pool_type: PoolType = PoolType.Conv2D