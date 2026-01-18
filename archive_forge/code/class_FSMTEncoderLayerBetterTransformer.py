from typing import TYPE_CHECKING
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.activations import ACT2FN
from .base import BetterTransformerBaseLayer
class FSMTEncoderLayerBetterTransformer(BetterTransformerBaseLayer, nn.Module):

    def __init__(self, fsmt_layer, config):
        """
        A simple conversion of the FSMT Encoder layer to its `BetterTransformer` implementation.

        Args:
            fsmt_layer (`torch.nn.Module`):
                The original FSMT Layer where the weights needs to be retrieved.
        """
        super().__init__(config)
        super(BetterTransformerBaseLayer, self).__init__()
        self.in_proj_weight = nn.Parameter(torch.cat([fsmt_layer.self_attn.q_proj.weight, fsmt_layer.self_attn.k_proj.weight, fsmt_layer.self_attn.v_proj.weight]))
        self.in_proj_bias = nn.Parameter(torch.cat([fsmt_layer.self_attn.q_proj.bias, fsmt_layer.self_attn.k_proj.bias, fsmt_layer.self_attn.v_proj.bias]))
        self.out_proj_weight = fsmt_layer.self_attn.out_proj.weight
        self.out_proj_bias = fsmt_layer.self_attn.out_proj.bias
        self.linear1_weight = fsmt_layer.fc1.weight
        self.linear1_bias = fsmt_layer.fc1.bias
        self.linear2_weight = fsmt_layer.fc2.weight
        self.linear2_bias = fsmt_layer.fc2.bias
        self.norm1_eps = fsmt_layer.self_attn_layer_norm.eps
        self.norm1_weight = fsmt_layer.self_attn_layer_norm.weight
        self.norm1_bias = fsmt_layer.self_attn_layer_norm.bias
        self.norm2_eps = fsmt_layer.final_layer_norm.eps
        self.norm2_weight = fsmt_layer.final_layer_norm.weight
        self.norm2_bias = fsmt_layer.final_layer_norm.bias
        self.num_heads = fsmt_layer.self_attn.num_heads
        self.embed_dim = fsmt_layer.self_attn.embed_dim
        self.is_last_layer = False
        self.original_layers_mapping = {'in_proj_weight': ['self_attn.q_proj.weight', 'self_attn.k_proj.weight', 'self_attn.v_proj.weight'], 'in_proj_bias': ['self_attn.q_proj.bias', 'self_attn.k_proj.bias', 'self_attn.v_proj.bias'], 'out_proj_weight': 'self_attn.out_proj.weight', 'out_proj_bias': 'self_attn.out_proj.bias', 'linear1_weight': 'fc1.weight', 'linear1_bias': 'fc1.bias', 'linear2_weight': 'fc2.weight', 'linear2_bias': 'fc2.bias', 'norm1_weight': 'self_attn_layer_norm.weight', 'norm1_bias': 'self_attn_layer_norm.bias', 'norm2_weight': 'final_layer_norm.weight', 'norm2_bias': 'final_layer_norm.bias'}
        self.validate_bettertransformer()

    def forward(self, hidden_states, attention_mask, output_attentions: bool, position_bias=None, *_, **__):
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
                attention_mask = attention_mask.bool()
                attention_mask = torch.reshape(attention_mask, (attention_mask.shape[0], attention_mask.shape[-1]))
                if hidden_states.shape[0] != attention_mask.shape[0]:
                    hidden_states = hidden_states.transpose(1, 0)
                    original_shape = hidden_states.shape
                hidden_states = torch._nested_tensor_from_mask(hidden_states, ~attention_mask)
                attention_mask = None
            hidden_states = torch._transformer_encoder_layer_fwd(hidden_states, self.embed_dim, self.num_heads, self.in_proj_weight, self.in_proj_bias, self.out_proj_weight, self.out_proj_bias, self.use_gelu, self.norm_first, self.norm1_eps, self.norm1_weight, self.norm1_bias, self.norm2_weight, self.norm2_bias, self.linear1_weight, self.linear1_bias, self.linear2_weight, self.linear2_bias, attention_mask)
            if not self.is_last_layer:
                hidden_states.original_shape = original_shape
            elif hidden_states.is_nested and self.is_last_layer:
                hidden_states = hidden_states.to_padded_tensor(0.0, original_shape)
        else:
            raise NotImplementedError('Training and Autocast are not implemented for BetterTransformer + FSMT. Please open an issue.')
        return (hidden_states, attention_mask)