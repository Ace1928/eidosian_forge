import copy
import math
import warnings
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np
import torch
from torch import Tensor, nn
from torch.cuda.amp import autocast
from ...activations import ACT2FN
from ...modeling_outputs import BaseModelOutput
from ...modeling_utils import PreTrainedModel
from ...utils import (
from ...utils.backbone_utils import load_backbone
from .configuration_oneformer import OneFormerConfig
class OneFormerTransformerDecoderFFNLayer(nn.Module):

    def __init__(self, d_model, dim_feedforward=2048, dropout=0.0, activation='relu', normalize_before=False, layer_norm_eps=1e-05):
        super().__init__()
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.activation = ACT2FN[activation]
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, output):
        output2 = self.linear2(self.dropout(self.activation(self.linear1(output))))
        output = output + self.dropout(output2)
        output = self.norm(output)
        return output

    def forward_pre(self, output):
        output2 = self.norm(output)
        output2 = self.linear2(self.dropout(self.activation(self.linear1(output2))))
        output = output + self.dropout(output2)
        return output

    def forward(self, output):
        if self.normalize_before:
            return self.forward_pre(output)
        return self.forward_post(output)