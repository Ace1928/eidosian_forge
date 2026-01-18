import math
from typing import Dict, List, Optional, Set, Tuple, Union
import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from ...activations import ACT2FN
from ...modeling_outputs import BaseModelOutput, BaseModelOutputWithPooling, SequenceClassifierOutput
from ...modeling_utils import PreTrainedModel
from ...pytorch_utils import find_pruneable_heads_and_indices, prune_linear_layer
from ...utils import add_code_sample_docstrings, add_start_docstrings, add_start_docstrings_to_model_forward, logging
from .configuration_audio_spectrogram_transformer import ASTConfig
class ASTMLPHead(nn.Module):

    def __init__(self, config: ASTConfig):
        super().__init__()
        self.layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dense = nn.Linear(config.hidden_size, config.num_labels) if config.num_labels > 0 else nn.Identity()

    def forward(self, hidden_state):
        hidden_state = self.layernorm(hidden_state)
        hidden_state = self.dense(hidden_state)
        return hidden_state