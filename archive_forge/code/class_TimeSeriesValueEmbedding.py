from typing import List, Optional, Tuple, Union
import numpy as np
import torch
from torch import nn
from ...activations import ACT2FN
from ...modeling_attn_mask_utils import _prepare_4d_attention_mask, _prepare_4d_causal_attention_mask
from ...modeling_outputs import (
from ...modeling_utils import PreTrainedModel
from ...time_series_utils import NegativeBinomialOutput, NormalOutput, StudentTOutput
from ...utils import (
from .configuration_time_series_transformer import TimeSeriesTransformerConfig
class TimeSeriesValueEmbedding(nn.Module):

    def __init__(self, feature_size, d_model):
        super().__init__()
        self.value_projection = nn.Linear(in_features=feature_size, out_features=d_model, bias=False)

    def forward(self, x):
        return self.value_projection(x)