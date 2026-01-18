import collections.abc
import math
from dataclasses import dataclass
from typing import Optional, Tuple
import torch
from torch import Tensor, nn
from ...activations import ACT2FN
from ...file_utils import ModelOutput
from ...modeling_outputs import BackboneOutput
from ...modeling_utils import PreTrainedModel
from ...pytorch_utils import find_pruneable_heads_and_indices, meshgrid, prune_linear_layer
from ...utils.backbone_utils import BackboneMixin
from .configuration_maskformer_swin import MaskFormerSwinConfig
class MaskFormerSwinEncoder(nn.Module):

    def __init__(self, config, grid_size):
        super().__init__()
        self.num_layers = len(config.depths)
        self.config = config
        dpr = [x.item() for x in torch.linspace(0, config.drop_path_rate, sum(config.depths))]
        self.layers = nn.ModuleList([MaskFormerSwinStage(config=config, dim=int(config.embed_dim * 2 ** i_layer), input_resolution=(grid_size[0] // 2 ** i_layer, grid_size[1] // 2 ** i_layer), depth=config.depths[i_layer], num_heads=config.num_heads[i_layer], drop_path=dpr[sum(config.depths[:i_layer]):sum(config.depths[:i_layer + 1])], downsample=MaskFormerSwinPatchMerging if i_layer < self.num_layers - 1 else None) for i_layer in range(self.num_layers)])
        self.gradient_checkpointing = False

    def forward(self, hidden_states, input_dimensions, head_mask=None, output_attentions=False, output_hidden_states=False, return_dict=True):
        all_hidden_states = () if output_hidden_states else None
        all_input_dimensions = ()
        all_self_attentions = () if output_attentions else None
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)
        for i, layer_module in enumerate(self.layers):
            layer_head_mask = head_mask[i] if head_mask is not None else None
            if self.gradient_checkpointing and self.training:
                layer_hidden_states, output_dimensions, layer_all_hidden_states = self._gradient_checkpointing_func(layer_module.__call__, hidden_states, layer_head_mask, output_attentions)
            else:
                layer_hidden_states, output_dimensions, layer_all_hidden_states = layer_module(hidden_states, input_dimensions, layer_head_mask, output_attentions, output_hidden_states)
            input_dimensions = (output_dimensions[-2], output_dimensions[-1])
            all_input_dimensions += (input_dimensions,)
            if output_hidden_states:
                all_hidden_states += (layer_all_hidden_states,)
            hidden_states = layer_hidden_states
            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_all_hidden_states[1],)
        if not return_dict:
            return tuple((v for v in [hidden_states, all_hidden_states, all_self_attentions] if v is not None))
        return MaskFormerSwinBaseModelOutput(last_hidden_state=hidden_states, hidden_states=all_hidden_states, hidden_states_spatial_dimensions=all_input_dimensions, attentions=all_self_attentions)