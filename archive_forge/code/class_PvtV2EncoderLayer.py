import math
from typing import Optional, Tuple, Union
import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from ...activations import ACT2FN
from ...modeling_outputs import BackboneOutput, BaseModelOutput, ImageClassifierOutput
from ...modeling_utils import PreTrainedModel
from ...pytorch_utils import find_pruneable_heads_and_indices, prune_linear_layer
from ...utils import (
from ...utils.backbone_utils import BackboneMixin
from .configuration_pvt_v2 import PvtV2Config
class PvtV2EncoderLayer(nn.Module):

    def __init__(self, config: PvtV2Config, layer_idx: int):
        super().__init__()
        self.patch_embedding = PvtV2OverlapPatchEmbeddings(config=config, layer_idx=layer_idx)
        drop_path_decays = torch.linspace(0, config.drop_path_rate, sum(config.depths)).tolist()
        block_layers = []
        for block_idx in range(config.depths[layer_idx]):
            block_layers.append(PvtV2BlockLayer(config=config, layer_idx=layer_idx, drop_path=drop_path_decays[sum(config.depths[:layer_idx]) + block_idx]))
        self.blocks = nn.ModuleList(block_layers)
        self.layer_norm = nn.LayerNorm(config.hidden_sizes[layer_idx], eps=config.layer_norm_eps)

    def forward(self, hidden_states, output_attentions):
        all_self_attentions = () if output_attentions else None
        hidden_states, height, width = self.patch_embedding(hidden_states)
        for block in self.blocks:
            layer_outputs = block(hidden_states, height, width, output_attentions)
            hidden_states = layer_outputs[0]
            if output_attentions:
                all_self_attentions += (layer_outputs[1],)
        hidden_states = self.layer_norm(hidden_states)
        outputs = (hidden_states,)
        if output_attentions:
            outputs += (all_self_attentions,)
        return (outputs, height, width)