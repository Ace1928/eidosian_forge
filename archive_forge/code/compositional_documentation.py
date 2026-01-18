import math
from dataclasses import dataclass
from typing import Optional
import torch
import torch.nn.functional as F
from torch import Tensor, nn
from xformers.components.attention import (
from xformers.components.attention.core import _softmax
from xformers.components.input_projection import InputProjection, InputProjectionConfig

        Input shape: Time x Batch x Channel

        Args:
            att_mask (ByteTensor, optional): typically used to
                implement causal attention, where the mask prevents the
                attention from looking forward in time (default: None).
        