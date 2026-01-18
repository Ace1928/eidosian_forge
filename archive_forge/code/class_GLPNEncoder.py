import math
from typing import List, Optional, Tuple, Union
import torch
import torch.utils.checkpoint
from torch import nn
from ...activations import ACT2FN
from ...modeling_outputs import BaseModelOutput, DepthEstimatorOutput
from ...modeling_utils import PreTrainedModel
from ...pytorch_utils import find_pruneable_heads_and_indices, prune_linear_layer
from ...utils import (
from .configuration_glpn import GLPNConfig
class GLPNEncoder(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config
        dpr = [x.item() for x in torch.linspace(0, config.drop_path_rate, sum(config.depths))]
        embeddings = []
        for i in range(config.num_encoder_blocks):
            embeddings.append(GLPNOverlapPatchEmbeddings(patch_size=config.patch_sizes[i], stride=config.strides[i], num_channels=config.num_channels if i == 0 else config.hidden_sizes[i - 1], hidden_size=config.hidden_sizes[i]))
        self.patch_embeddings = nn.ModuleList(embeddings)
        blocks = []
        cur = 0
        for i in range(config.num_encoder_blocks):
            layers = []
            if i != 0:
                cur += config.depths[i - 1]
            for j in range(config.depths[i]):
                layers.append(GLPNLayer(config, hidden_size=config.hidden_sizes[i], num_attention_heads=config.num_attention_heads[i], drop_path=dpr[cur + j], sequence_reduction_ratio=config.sr_ratios[i], mlp_ratio=config.mlp_ratios[i]))
            blocks.append(nn.ModuleList(layers))
        self.block = nn.ModuleList(blocks)
        self.layer_norm = nn.ModuleList([nn.LayerNorm(config.hidden_sizes[i]) for i in range(config.num_encoder_blocks)])

    def forward(self, pixel_values, output_attentions=False, output_hidden_states=False, return_dict=True):
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None
        batch_size = pixel_values.shape[0]
        hidden_states = pixel_values
        for idx, x in enumerate(zip(self.patch_embeddings, self.block, self.layer_norm)):
            embedding_layer, block_layer, norm_layer = x
            hidden_states, height, width = embedding_layer(hidden_states)
            for i, blk in enumerate(block_layer):
                layer_outputs = blk(hidden_states, height, width, output_attentions)
                hidden_states = layer_outputs[0]
                if output_attentions:
                    all_self_attentions = all_self_attentions + (layer_outputs[1],)
            hidden_states = norm_layer(hidden_states)
            hidden_states = hidden_states.reshape(batch_size, height, width, -1).permute(0, 3, 1, 2).contiguous()
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)
        if not return_dict:
            return tuple((v for v in [hidden_states, all_hidden_states, all_self_attentions] if v is not None))
        return BaseModelOutput(last_hidden_state=hidden_states, hidden_states=all_hidden_states, attentions=all_self_attentions)