import math
from typing import List, Optional, Tuple, Union
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from ...activations import ACT2FN
from ...modeling_outputs import (
from ...modeling_utils import PreTrainedModel
from ...pytorch_utils import ALL_LAYERNORM_LAYERS
from ...utils import (
from .configuration_mega import MegaConfig
class MegaNormalizedFeedForwardNetwork(nn.Module):
    """
    Normalized feed-forward network used in Mega blocks. Left as-is from original Mega repo aside from retrieving args
    from Hugging Face config
    """

    def __init__(self, config: MegaConfig):
        super().__init__()
        self.config = config
        self.hidden_dim = config.nffn_hidden_size
        self.act_fn = config.activation
        self.activation = ACT2FN[config.activation]
        self.dropout = MegaDropout(self.config.dropout_prob, is_featurewise=self.config.use_feature_dropout)
        self.hidden_dropout = MegaDropout(self.config.nffn_activation_dropout_prob, is_featurewise=self.config.use_feature_dropout)
        self.prenorm = self.config.normalize_before_ffn
        self.norm = MegaSequenceNorm(self.config.normalization_type, self.config.hidden_size, affine=self.config.norm_affine)
        self.fc1 = nn.Linear(self.config.hidden_size, self.config.nffn_hidden_size)
        self.fc2 = nn.Linear(self.config.nffn_hidden_size, self.config.hidden_size)

    def forward(self, inputs):
        residual = inputs
        if self.prenorm:
            inputs = self.norm(inputs)
        hidden = self.activation(self.fc1(inputs))
        hidden = self.hidden_dropout(hidden)
        output = self.fc2(hidden)
        output = self.dropout(output)
        output = output + residual
        if not self.prenorm:
            output = self.norm(output)
        return output