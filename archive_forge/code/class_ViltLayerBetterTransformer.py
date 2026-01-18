from typing import TYPE_CHECKING
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.activations import ACT2FN
from .base import BetterTransformerBaseLayer
class ViltLayerBetterTransformer(BetterTransformerBaseLayer, nn.Module):

    def __init__(self, vilt_layer, config):
        """
        A simple conversion of the VilTLayer to its `BetterTransformer` implementation.

        Args:
            vilt_layer (`torch.nn.Module`):
                The original `VilTLayer` where the weights needs to be retrieved.
        """
        super().__init__(config)
        super(BetterTransformerBaseLayer, self).__init__()
        self.in_proj_weight = nn.Parameter(torch.cat([vilt_layer.attention.attention.query.weight, vilt_layer.attention.attention.key.weight, vilt_layer.attention.attention.value.weight]))
        self.in_proj_bias = nn.Parameter(torch.cat([vilt_layer.attention.attention.query.bias, vilt_layer.attention.attention.key.bias, vilt_layer.attention.attention.value.bias]))
        self.out_proj_weight = vilt_layer.attention.output.dense.weight
        self.out_proj_bias = vilt_layer.attention.output.dense.bias
        self.linear1_weight = vilt_layer.intermediate.dense.weight
        self.linear1_bias = vilt_layer.intermediate.dense.bias
        self.linear2_weight = vilt_layer.output.dense.weight
        self.linear2_bias = vilt_layer.output.dense.bias
        self.norm1_eps = vilt_layer.layernorm_before.eps
        self.norm1_weight = vilt_layer.layernorm_before.weight
        self.norm1_bias = vilt_layer.layernorm_before.bias
        self.norm2_eps = vilt_layer.layernorm_after.eps
        self.norm2_weight = vilt_layer.layernorm_after.weight
        self.norm2_bias = vilt_layer.layernorm_after.bias
        self.num_heads = vilt_layer.attention.attention.num_attention_heads
        self.embed_dim = int(vilt_layer.attention.attention.attention_head_size * self.num_heads)
        self.is_last_layer = False
        self.norm_first = True
        self.original_layers_mapping = {'in_proj_weight': ['attention.attention.query.weight', 'attention.attention.key.weight', 'attention.attention.value.weight'], 'in_proj_bias': ['attention.attention.query.bias', 'attention.attention.key.bias', 'attention.attention.value.bias'], 'out_proj_weight': 'attention.output.dense.weight', 'out_proj_bias': 'attention.output.dense.bias', 'linear1_weight': 'intermediate.dense.weight', 'linear1_bias': 'intermediate.dense.bias', 'linear2_weight': 'output.dense.weight', 'linear2_bias': 'output.dense.bias', 'norm1_weight': 'layernorm_before.weight', 'norm1_bias': 'layernorm_before.bias', 'norm2_weight': 'layernorm_after.weight', 'norm2_bias': 'layernorm_after.bias'}
        self.validate_bettertransformer()

    def forward(self, hidden_states, layer_head_mask, output_attentions: bool, *_, **__):
        if output_attentions:
            raise ValueError('output_attentions=True can not be supported with BetterTransformer.')
        if not self.training and (not torch.is_autocast_enabled()) and (not torch.is_autocast_cpu_enabled()):
            attention_mask = None
            hidden_states = torch._transformer_encoder_layer_fwd(hidden_states, self.embed_dim, self.num_heads, self.in_proj_weight, self.in_proj_bias, self.out_proj_weight, self.out_proj_bias, self.use_gelu, self.norm_first, self.norm1_eps, self.norm1_weight, self.norm1_bias, self.norm2_weight, self.norm2_bias, self.linear1_weight, self.linear1_bias, self.linear2_weight, self.linear2_bias, attention_mask)
            if hidden_states.is_nested and self.is_last_layer:
                hidden_states = hidden_states.to_padded_tensor(0.0)
        else:
            raise NotImplementedError('Training and Autocast are not implemented for BetterTransformer + Vilt. Please open an issue.')
        return (hidden_states,)