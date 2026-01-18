import collections
import logging
import math
import random
from abc import ABC, abstractmethod
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence, Tuple, Union
import torch
from torch import Tensor, nn
from torch.nn import CrossEntropyLoss
from transformers import UdopConfig
from transformers.modeling_outputs import (
from ...activations import ACT2FN
from ...modeling_utils import PreTrainedModel
from ...pytorch_utils import find_pruneable_heads_and_indices, prune_linear_layer
from ...utils import (
from ..deprecated._archive_maps import UDOP_PRETRAINED_MODEL_ARCHIVE_LIST  # noqa: F401, E402
class UdopLayerFF(nn.Module):

    def __init__(self, config: UdopConfig):
        super().__init__()
        if config.is_gated_act:
            self.DenseReluDense = UdopDenseGatedActDense(config)
        else:
            self.DenseReluDense = UdopDenseActDense(config)
        self.layer_norm = UdopLayerNorm(config.d_model, eps=config.layer_norm_epsilon)
        self.dropout = nn.Dropout(config.dropout_rate)

    def forward(self, hidden_states):
        forwarded_states = self.layer_norm(hidden_states)
        forwarded_states = self.DenseReluDense(forwarded_states)
        hidden_states = hidden_states + self.dropout(forwarded_states)
        return hidden_states