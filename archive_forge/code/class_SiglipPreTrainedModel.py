import math
import warnings
from dataclasses import dataclass
from typing import Any, Optional, Tuple, Union
import numpy as np
import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from torch.nn.init import _calculate_fan_in_and_fan_out
from ...activations import ACT2FN
from ...modeling_attn_mask_utils import _prepare_4d_attention_mask
from ...modeling_outputs import BaseModelOutput, BaseModelOutputWithPooling, ImageClassifierOutput
from ...modeling_utils import PreTrainedModel
from ...utils import (
from .configuration_siglip import SiglipConfig, SiglipTextConfig, SiglipVisionConfig
class SiglipPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """
    config_class = SiglipConfig
    base_model_prefix = 'siglip'
    supports_gradient_checkpointing = True

    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, SiglipVisionEmbeddings):
            width = self.config.vision_config.hidden_size if isinstance(self.config, SiglipConfig) else self.config.hidden_size
            nn.init.normal_(module.position_embedding.weight, std=1 / np.sqrt(width))
        elif isinstance(module, nn.Embedding):
            default_flax_embed_init(module.weight)
        elif isinstance(module, SiglipAttention):
            nn.init.xavier_uniform_(module.q_proj.weight)
            nn.init.xavier_uniform_(module.k_proj.weight)
            nn.init.xavier_uniform_(module.v_proj.weight)
            nn.init.xavier_uniform_(module.out_proj.weight)
            nn.init.zeros_(module.q_proj.bias)
            nn.init.zeros_(module.k_proj.bias)
            nn.init.zeros_(module.v_proj.bias)
            nn.init.zeros_(module.out_proj.bias)
        elif isinstance(module, SiglipMLP):
            nn.init.xavier_uniform_(module.fc1.weight)
            nn.init.xavier_uniform_(module.fc2.weight)
            nn.init.normal_(module.fc1.bias, std=1e-06)
            nn.init.normal_(module.fc2.bias, std=1e-06)
        elif isinstance(module, SiglipMultiheadAttentionPoolingHead):
            nn.init.xavier_uniform_(module.probe.data)
            nn.init.xavier_uniform_(module.attention.in_proj_weight.data)
            nn.init.zeros_(module.attention.in_proj_bias.data)
        elif isinstance(module, SiglipModel):
            logit_scale_init = torch.log(torch.tensor(1.0))
            module.logit_scale.data.fill_(logit_scale_init)
            module.logit_bias.data.zero_()
        elif isinstance(module, (nn.Linear, nn.Conv2d)):
            lecun_normal_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)