import collections.abc
import math
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union
import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from ...activations import ACT2FN
from ...modeling_outputs import (
from ...modeling_utils import PreTrainedModel
from ...pytorch_utils import find_pruneable_heads_and_indices, meshgrid, prune_linear_layer
from ...utils import (
from .configuration_data2vec_vision import Data2VecVisionConfig
class Data2VecVisionLayer(nn.Module):
    """This corresponds to the Block class in the timm implementation."""

    def __init__(self, config: Data2VecVisionConfig, window_size: Optional[tuple]=None, drop_path_rate: float=0.0) -> None:
        super().__init__()
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.seq_len_dim = 1
        self.attention = Data2VecVisionAttention(config, window_size=window_size)
        self.intermediate = Data2VecVisionIntermediate(config)
        self.output = Data2VecVisionOutput(config)
        self.layernorm_before = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.drop_path = Data2VecVisionDropPath(drop_path_rate) if drop_path_rate > 0.0 else nn.Identity()
        self.layernorm_after = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        init_values = config.layer_scale_init_value
        if init_values > 0:
            self.lambda_1 = nn.Parameter(init_values * torch.ones(config.hidden_size), requires_grad=True)
            self.lambda_2 = nn.Parameter(init_values * torch.ones(config.hidden_size), requires_grad=True)
        else:
            self.lambda_1, self.lambda_2 = (None, None)

    def forward(self, hidden_states: torch.Tensor, head_mask: Optional[torch.Tensor]=None, output_attentions: bool=False, relative_position_bias: Optional['Data2VecVisionRelativePositionBias']=None) -> Union[Tuple[torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]:
        self_attention_outputs = self.attention(self.layernorm_before(hidden_states), head_mask, output_attentions=output_attentions, relative_position_bias=relative_position_bias)
        attention_output = self_attention_outputs[0]
        outputs = self_attention_outputs[1:]
        if self.lambda_1 is not None:
            attention_output = self.lambda_1 * attention_output
        hidden_states = self.drop_path(attention_output) + hidden_states
        layer_output = self.layernorm_after(hidden_states)
        layer_output = self.intermediate(layer_output)
        layer_output = self.output(layer_output)
        if self.lambda_2 is not None:
            layer_output = self.lambda_2 * layer_output
        layer_output = self.drop_path(layer_output) + hidden_states
        outputs = (layer_output,) + outputs
        return outputs