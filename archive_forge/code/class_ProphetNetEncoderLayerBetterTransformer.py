from typing import TYPE_CHECKING
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.activations import ACT2FN
from .base import BetterTransformerBaseLayer
class ProphetNetEncoderLayerBetterTransformer(BetterTransformerBaseLayer, nn.Module):

    def __init__(self, prophetnet_layer, config):
        """
        A simple conversion of the ProphetNet Encoder layer to its `BetterTransformer` implementation.

        Args:
            prophet_net_layer (`torch.nn.Module`):
                The original ProphetNet Layer where the weights needs to be retrieved.
        """
        super().__init__(config)
        super(BetterTransformerBaseLayer, self).__init__()
        self.config = config
        self.in_proj_weight = nn.Parameter(torch.cat([prophetnet_layer.self_attn.query_proj.weight, prophetnet_layer.self_attn.key_proj.weight, prophetnet_layer.self_attn.value_proj.weight]))
        self.in_proj_bias = nn.Parameter(torch.cat([prophetnet_layer.self_attn.query_proj.bias, prophetnet_layer.self_attn.key_proj.bias, prophetnet_layer.self_attn.value_proj.bias]))
        self.out_proj_weight = prophetnet_layer.self_attn.out_proj.weight
        self.out_proj_bias = prophetnet_layer.self_attn.out_proj.bias
        self.linear1_weight = prophetnet_layer.feed_forward.intermediate.weight
        self.linear1_bias = prophetnet_layer.feed_forward.intermediate.bias
        self.linear2_weight = prophetnet_layer.feed_forward.output.weight
        self.linear2_bias = prophetnet_layer.feed_forward.output.bias
        self.norm1_eps = prophetnet_layer.self_attn_layer_norm.eps
        self.norm1_weight = prophetnet_layer.self_attn_layer_norm.weight
        self.norm1_bias = prophetnet_layer.self_attn_layer_norm.bias
        self.norm2_eps = prophetnet_layer.feed_forward_layer_norm.eps
        self.norm2_weight = prophetnet_layer.feed_forward_layer_norm.weight
        self.norm2_bias = prophetnet_layer.feed_forward_layer_norm.bias
        self.num_heads = prophetnet_layer.self_attn.num_attn_heads
        self.embed_dim = prophetnet_layer.self_attn.head_dim * self.num_heads
        self.is_last_layer = False
        self.original_layers_mapping = {'in_proj_weight': ['self_attn.query_proj.weight', 'self_attn.key_proj.weight', 'self_attn.value_proj.weight'], 'in_proj_bias': ['self_attn.query_proj.bias', 'self_attn.key_proj.bias', 'self_attn.value_proj.bias'], 'out_proj_weight': 'self_attn.out_proj.weight', 'out_proj_bias': 'self_attn.out_proj.bias', 'linear1_weight': 'feed_forward.intermediate.weight', 'linear1_bias': 'feed_forward.intermediate.bias', 'linear2_weight': 'feed_forward.output.weight', 'linear2_bias': 'feed_forward.output.bias', 'norm1_weight': 'self_attn_layer_norm.weight', 'norm1_bias': 'self_attn_layer_norm.bias', 'norm2_weight': 'feed_forward_layer_norm.weight', 'norm2_bias': 'feed_forward_layer_norm.bias'}
        self.validate_bettertransformer()

    def forward(self, hidden_states, attention_mask, output_attentions: bool, *_, **__):
        if output_attentions:
            raise ValueError('output_attentions=True can not be supported with BetterTransformer.')
        if not self.training and (not torch.is_autocast_enabled()) and (not torch.is_autocast_cpu_enabled()):
            if not hasattr(hidden_states, 'original_shape'):
                original_shape = hidden_states.shape
            else:
                original_shape = hidden_states.original_shape
            if hidden_states.is_nested:
                attention_mask = None
            if attention_mask is not None:
                attention_mask = attention_mask.squeeze(1)[:, 0]
                attention_mask = attention_mask.bool()
                attention_mask = torch.reshape(attention_mask, (attention_mask.shape[0], attention_mask.shape[-1]))
                hidden_states = torch._nested_tensor_from_mask(hidden_states, ~attention_mask)
                attention_mask = None
            hidden_states = torch._transformer_encoder_layer_fwd(hidden_states, self.embed_dim, self.num_heads, self.in_proj_weight, self.in_proj_bias, self.out_proj_weight, self.out_proj_bias, self.use_gelu, self.norm_first, self.norm1_eps, self.norm1_weight, self.norm1_bias, self.norm2_weight, self.norm2_bias, self.linear1_weight, self.linear1_bias, self.linear2_weight, self.linear2_bias, attention_mask)
            if not self.is_last_layer:
                hidden_states.original_shape = original_shape
            elif hidden_states.is_nested and self.is_last_layer:
                hidden_states = hidden_states.to_padded_tensor(0.0, original_shape)
        else:
            raise ValueError('Training and Autocast are not implemented for BetterTransformer + ProphetNet. Please open an issue.')
        return (hidden_states,)