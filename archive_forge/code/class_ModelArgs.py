import math
from dataclasses import dataclass
from typing import Any, Optional, Tuple
import fairscale.nn.model_parallel.initialize as fs_init
import torch
import torch.nn.functional as F
from fairscale.nn.model_parallel.layers import (
from torch import nn
import json
import os
import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple, TypedDict
from typing import Optional
import types, gc, os, time, re
import torch
import torch.nn as nn
from torch.nn import functional as F
@dataclass
class ModelArgs:
    """Data class for storing model arguments with default values and optional types.

    Attributes:
        dim (int): Dimensionality of the model or layers.
        n_layers (int): Number of layers in the model.
        n_heads (int): Number of attention heads in each layer.
        n_kv_heads (Optional[int]): Number of key/value heads, if different from n_heads.
        vocab_size (int): Size of the vocabulary. Set later by tokenizer.
        multiple_of (int): Ensures certain dimensions are multiples of this value for efficiency.
        ffn_dim_multiplier (Optional[float]): Multiplier for the dimension of feed-forward network.
        norm_eps (float): Epsilon value for normalization layers to avoid division by zero.
        rope_theta (float): Theta value for rotary position embeddings.
        max_batch_size (int): Maximum batch size that can be processed.
        max_seq_len (int): Maximum sequence length the model can handle.
    """
    dim: int = 4096
    n_layers: int = 32
    n_heads: int = 32
    n_kv_heads: Optional[int] = None
    vocab_size: int = -1
    multiple_of: int = 256
    ffn_dim_multiplier: Optional[float] = None
    norm_eps: float = 1e-05
    rope_theta: float = 10000
    max_batch_size: int = 32
    max_seq_len: int = 2048