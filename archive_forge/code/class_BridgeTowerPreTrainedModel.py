import math
from collections import OrderedDict
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union
import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import CrossEntropyLoss
from ...activations import ACT2FN, QuickGELUActivation
from ...modeling_outputs import (
from ...modeling_utils import PreTrainedModel, apply_chunking_to_forward
from ...pytorch_utils import find_pruneable_heads_and_indices, prune_linear_layer
from ...utils import add_start_docstrings, add_start_docstrings_to_model_forward, logging, replace_return_docstrings
from .configuration_bridgetower import BridgeTowerConfig, BridgeTowerTextConfig, BridgeTowerVisionConfig
class BridgeTowerPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """
    config_class = BridgeTowerConfig
    base_model_prefix = 'bridgetower'
    supports_gradient_checkpointing = False
    _no_split_modules = ['BridgeTowerSelfAttention', 'BridgeTowerResidualAttention']
    _skip_keys_device_placement = 'past_key_values'

    def _init_weights(self, module):
        if isinstance(module, BridgeTowerVisionModel):
            proj_std = module.visual.transformer.hidden_size ** (-0.5) * (2 * module.visual.transformer.num_hidden_layers) ** (-0.5)
            attn_std = module.visual.transformer.hidden_size ** (-0.5)
            fc_std = (2 * module.visual.transformer.hidden_size) ** (-0.5)
            for block in module.visual.transformer.resblocks:
                nn.init.normal_(block.attn.in_proj_weight, std=attn_std * self.config.initializer_factor)
                nn.init.normal_(block.attn.out_proj.weight, std=proj_std * self.config.initializer_factor)
                nn.init.normal_(block.mlp.c_fc.weight, std=fc_std * self.config.initializer_factor)
                nn.init.normal_(block.mlp.c_proj.weight, std=proj_std * self.config.initializer_factor)
            nn.init.normal_(module.visual.embeddings.class_embedding, std=attn_std * self.config.initializer_factor)
            nn.init.normal_(module.visual.embeddings.position_embedding.weight, std=attn_std * self.config.initializer_factor)
        elif isinstance(module, (nn.Linear, nn.Conv2d, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.05 * self.config.initializer_factor)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()