import math
from dataclasses import dataclass
import torch
import torch.nn as nn
from xformers.components.attention import Attention, AttentionConfig, register_attention
@dataclass
class VisualAttentionConfig(AttentionConfig):
    dim_model: int