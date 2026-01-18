import math
from dataclasses import dataclass
from typing import Any, List, Optional, Tuple, Union
import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import CrossEntropyLoss
from ...activations import ACT2FN
from ...modeling_outputs import (
from ...modeling_utils import PreTrainedModel
from ...utils import (
from .configuration_kosmos2 import Kosmos2Config, Kosmos2TextConfig, Kosmos2VisionConfig
class Kosmos2PreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """
    config_class = Kosmos2Config
    supports_gradient_checkpointing = True
    _no_split_modules = ['Kosmos2VisionEncoderLayer', 'Kosmos2TextBlock']

    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(self, Kosmos2VisionModel):
            factor = self.config.initializer_factor
        elif isinstance(self, (Kosmos2Model, Kosmos2ForConditionalGeneration)):
            factor = self.config.vision_config.initializer_factor
        if isinstance(self, (Kosmos2TextModel, Kosmos2TextForCausalLM)):
            std = self.config.init_std
        elif isinstance(self, (Kosmos2Model, Kosmos2ForConditionalGeneration)):
            std = self.config.text_config.init_std
        if isinstance(module, Kosmos2VisionEmbeddings):
            nn.init.normal_(module.class_embedding, mean=0.0, std=module.embed_dim ** (-0.5) * factor)
            nn.init.normal_(module.patch_embedding.weight, std=module.config.initializer_range * factor)
            nn.init.normal_(module.position_embedding.weight, std=module.config.initializer_range * factor)
        elif isinstance(module, Kosmos2VisionAttention):
            in_proj_std = module.embed_dim ** (-0.5) * (2 * module.config.num_hidden_layers) ** (-0.5) * factor
            out_proj_std = module.embed_dim ** (-0.5) * factor
            nn.init.normal_(module.q_proj.weight, std=in_proj_std)
            nn.init.normal_(module.k_proj.weight, std=in_proj_std)
            nn.init.normal_(module.v_proj.weight, std=in_proj_std)
            nn.init.normal_(module.out_proj.weight, std=out_proj_std)
            if module.q_proj.bias is not None:
                module.q_proj.bias.data.zero_()
            if module.k_proj.bias is not None:
                module.k_proj.bias.data.zero_()
            if module.v_proj.bias is not None:
                module.v_proj.bias.data.zero_()
            if module.out_proj.bias is not None:
                module.out_proj.bias.data.zero_()
        elif isinstance(module, Kosmos2VisionMLP):
            in_proj_std = module.config.hidden_size ** (-0.5) * (2 * module.config.num_hidden_layers) ** (-0.5) * factor
            fc_std = (2 * module.config.hidden_size) ** (-0.5) * factor
            nn.init.normal_(module.fc1.weight, std=fc_std)
            nn.init.normal_(module.fc2.weight, std=in_proj_std)
            if module.fc1.bias is not None:
                module.fc1.bias.data.zero_()
            if module.fc2.bias is not None:
                module.fc2.bias.data.zero_()
        elif isinstance(module, Kosmos2VisionEncoderLayer):
            module.layer_norm1.bias.data.zero_()
            module.layer_norm1.weight.data.fill_(1.0)
            module.layer_norm2.bias.data.zero_()
            module.layer_norm2.weight.data.fill_(1.0)
        elif isinstance(module, Kosmos2VisionTransformer):
            module.pre_layrnorm.bias.data.zero_()
            module.pre_layrnorm.weight.data.fill_(1.0)
            module.post_layernorm.bias.data.zero_()
            module.post_layernorm.weight.data.fill_(1.0)
        elif isinstance(module, KosmosTextAttention):
            nn.init.normal_(module.q_proj.weight, std=std)
            nn.init.normal_(module.k_proj.weight, std=std)
            nn.init.normal_(module.v_proj.weight, std=std)
            nn.init.normal_(module.out_proj.weight, std=std)
            if module.q_proj.bias is not None:
                module.q_proj.bias.data.zero_()
            if module.k_proj.bias is not None:
                module.k_proj.bias.data.zero_()
            if module.v_proj.bias is not None:
                module.v_proj.bias.data.zero_()
            if module.out_proj.bias is not None:
                module.out_proj.bias.data.zero_()
        elif isinstance(module, Kosmos2TextFFN):
            nn.init.normal_(module.fc1.weight, std=std)
            nn.init.normal_(module.fc2.weight, std=std)
            if module.fc1.bias is not None:
                module.fc1.bias.data.zero_()
            if module.fc2.bias is not None:
                module.fc2.bias.data.zero_()
        elif isinstance(module, Kosmos2TextForCausalLM):
            nn.init.normal_(module.lm_head.weight, std=std)
            if module.lm_head.bias is not None:
                module.lm_head.bias.data.zero_()
        elif isinstance(module, Kosmos2ImageToTextProjection):
            nn.init.normal_(module.dense.weight, std=std)
            if module.dense.bias is not None:
                module.dense.bias.data.zero_()
        elif isinstance(module, Kosmos2TextTransformer):
            module.embed_tokens.weight.data.normal_(mean=0.0, std=std)
            if module.embed_tokens.padding_idx is not None:
                module.embed_tokens.weight.data[module.embed_tokens.padding_idx].zero_()