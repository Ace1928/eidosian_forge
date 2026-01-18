import collections
import math
from dataclasses import dataclass
from typing import Any, List, Optional, Tuple, Union
import torch
import torch.nn.functional as F
from torch import nn
from ...activations import ACT2FN
from ...modeling_outputs import (
from ...modeling_utils import PreTrainedModel
from ...pytorch_utils import apply_chunking_to_forward, find_pruneable_heads_and_indices, meshgrid, prune_linear_layer
from ...utils import (
from .configuration_clap import ClapAudioConfig, ClapConfig, ClapTextConfig
class ClapPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """
    config_class = ClapConfig
    base_model_prefix = 'clap'
    supports_gradient_checkpointing = False

    def _init_weights(self, module):
        """Initialize the weights"""
        factor = self.config.initializer_factor
        if isinstance(module, ClapTextEmbeddings):
            module.position_embeddings.weight.data.normal_(mean=0.0, std=factor * 0.02)
            module.token_type_embeddings.weight.data.normal_(mean=0.0, std=factor * 0.02)
        elif isinstance(module, ClapModel):
            nn.init.normal_(module.logit_scale_a, std=factor * 0.02)
            nn.init.normal_(module.logit_scale_t, std=factor * 0.02)
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=factor * 0.02)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        elif isinstance(module, (nn.Conv2d, nn.Linear)):
            in_proj_std = self.config.hidden_size ** (-0.5) * (2 * self.config.num_hidden_layers) ** (-0.5) * factor
            nn.init.normal_(module.weight, std=in_proj_std)
            if module.bias is not None:
                module.bias.data.zero_()