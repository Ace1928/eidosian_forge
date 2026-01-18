import math
from dataclasses import dataclass
from typing import Any, List, Optional, Tuple, Union
import torch
import torch.utils.checkpoint
from torch import nn
from ...activations import ACT2FN
from ...modeling_outputs import (
from ...modeling_utils import PreTrainedModel
from ...pytorch_utils import apply_chunking_to_forward, find_pruneable_heads_and_indices, prune_linear_layer
from ...utils import (
from .configuration_chinese_clip import ChineseCLIPConfig, ChineseCLIPTextConfig, ChineseCLIPVisionConfig
class ChineseCLIPPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """
    config_class = ChineseCLIPConfig
    base_model_prefix = 'chinese_clip'
    supports_gradient_checkpointing = True

    def _init_weights(self, module):
        """Initialize the weights"""
        factor = self.config.initializer_factor
        if isinstance(module, ChineseCLIPVisionEmbeddings):
            factor = self.config.initializer_factor
            nn.init.normal_(module.class_embedding, mean=0.0, std=module.embed_dim ** (-0.5) * factor)
            nn.init.normal_(module.patch_embedding.weight, std=module.config.initializer_range * factor)
            nn.init.normal_(module.position_embedding.weight, std=module.config.initializer_range * factor)
        elif isinstance(module, ChineseCLIPTextEmbeddings):
            nn.init.normal_(module.word_embeddings.weight, mean=0.0, std=self.config.initializer_range)
            nn.init.normal_(module.position_embeddings.weight, mean=0.0, std=self.config.initializer_range)
            nn.init.normal_(module.token_type_embeddings.weight, mean=0.0, std=self.config.initializer_range)
            for embedding in [module.word_embeddings, module.position_embeddings, module.token_type_embeddings]:
                if embedding.padding_idx is not None:
                    embedding.weight.data[embedding.padding_idx].zero_()
        elif isinstance(module, ChineseCLIPVisionAttention):
            factor = self.config.initializer_factor
            in_proj_std = module.embed_dim ** (-0.5) * (2 * module.config.num_hidden_layers) ** (-0.5) * factor
            out_proj_std = module.embed_dim ** (-0.5) * factor
            nn.init.normal_(module.q_proj.weight, std=in_proj_std)
            nn.init.normal_(module.k_proj.weight, std=in_proj_std)
            nn.init.normal_(module.v_proj.weight, std=in_proj_std)
            nn.init.normal_(module.out_proj.weight, std=out_proj_std)
        elif isinstance(module, ChineseCLIPVisionMLP):
            factor = self.config.initializer_factor
            in_proj_std = module.config.hidden_size ** (-0.5) * (2 * module.config.num_hidden_layers) ** (-0.5) * factor
            fc_std = (2 * module.config.hidden_size) ** (-0.5) * factor
            nn.init.normal_(module.fc1.weight, std=fc_std)
            nn.init.normal_(module.fc2.weight, std=in_proj_std)
        elif isinstance(module, ChineseCLIPModel):
            nn.init.normal_(module.text_projection.weight, std=module.text_embed_dim ** (-0.5) * self.config.initializer_factor)
            nn.init.normal_(module.visual_projection.weight, std=module.vision_embed_dim ** (-0.5) * self.config.initializer_factor)
        if isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()