import collections.abc
from typing import Optional, Tuple, Union
import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from ...activations import ACT2FN
from ...modeling_outputs import BaseModelOutputWithNoAttention, ImageClassifierOutputWithNoAttention
from ...modeling_utils import PreTrainedModel
from ...utils import add_code_sample_docstrings, add_start_docstrings, add_start_docstrings_to_model_forward, logging
from .configuration_poolformer import PoolFormerConfig
class PoolFormerLayer(nn.Module):
    """This corresponds to the 'PoolFormerBlock' class in the original implementation."""

    def __init__(self, config, num_channels, pool_size, hidden_size, intermediate_size, drop_path):
        super().__init__()
        self.pooling = PoolFormerPooling(pool_size)
        self.output = PoolFormerOutput(config, drop_path, hidden_size, intermediate_size)
        self.before_norm = PoolFormerGroupNorm(num_channels)
        self.after_norm = PoolFormerGroupNorm(num_channels)
        self.drop_path = PoolFormerDropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.use_layer_scale = config.use_layer_scale
        if config.use_layer_scale:
            self.layer_scale_1 = nn.Parameter(config.layer_scale_init_value * torch.ones(num_channels), requires_grad=True)
            self.layer_scale_2 = nn.Parameter(config.layer_scale_init_value * torch.ones(num_channels), requires_grad=True)

    def forward(self, hidden_states):
        if self.use_layer_scale:
            pooling_output = self.pooling(self.before_norm(hidden_states))
            scaled_op = self.layer_scale_1.unsqueeze(-1).unsqueeze(-1) * pooling_output
            hidden_states = hidden_states + self.drop_path(scaled_op)
            outputs = ()
            layer_output = self.output(self.after_norm(hidden_states))
            scaled_op = self.layer_scale_2.unsqueeze(-1).unsqueeze(-1) * layer_output
            output = hidden_states + self.drop_path(scaled_op)
            outputs = (output,) + outputs
            return outputs
        else:
            pooling_output = self.drop_path(self.pooling(self.before_norm(hidden_states)))
            hidden_states = pooling_output + hidden_states
            outputs = ()
            layer_output = self.drop_path(self.output(self.after_norm(hidden_states)))
            output = hidden_states + layer_output
            outputs = (output,) + outputs
            return outputs