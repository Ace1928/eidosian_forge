import logging
import math
from dataclasses import dataclass
from typing import Optional, Tuple
import torch
import torch.nn as nn
from torch.cuda.amp import autocast
from xformers.components.attention import Attention, AttentionConfig, register_attention
from xformers.components.attention.feature_maps import (
@dataclass
class FavorAttentionConfig(AttentionConfig):
    causal: Optional[bool]
    dim_features: Optional[int] = None
    dim_head: Optional[int] = None
    iter_before_redraw: Optional[int] = None
    feature_map: Optional[FeatureMapType] = None