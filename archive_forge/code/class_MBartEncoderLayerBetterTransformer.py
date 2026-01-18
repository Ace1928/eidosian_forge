from typing import TYPE_CHECKING
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.activations import ACT2FN
from .base import BetterTransformerBaseLayer
class MBartEncoderLayerBetterTransformer(BetterTransformerBaseLayer, nn.Module):

    def __init__(self, mbart_layer, config):
        """
        A simple conversion of the `MBartEncoderLayer` to its `BetterTransformer` implementation.
        Args:
            mbart_layer (`torch.nn.Module`):
                The original `MBartEncoderLayer` where the weights needs to be retrieved.
        """
        super().__init__(config)
        super(BetterTransformerBaseLayer, self).__init__()
        self.in_proj_weight = nn.Parameter(torch.cat([mbart_layer.self_attn.q_proj.weight, mbart_layer.self_attn.k_proj.weight, mbart_layer.self_attn.v_proj.weight]))
        self.in_proj_bias = nn.Parameter(torch.cat([mbart_layer.self_attn.q_proj.bias, mbart_layer.self_attn.k_proj.bias, mbart_layer.self_attn.v_proj.bias]))
        self.out_proj_weight = mbart_layer.self_attn.out_proj.weight
        self.out_proj_bias = mbart_layer.self_attn.out_proj.bias
        self.linear1_weight = mbart_layer.fc1.weight
        self.linear1_bias = mbart_layer.fc1.bias
        self.linear2_weight = mbart_layer.fc2.weight
        self.linear2_bias = mbart_layer.fc2.bias
        self.norm1_eps = mbart_layer.self_attn_layer_norm.eps
        self.norm1_weight = mbart_layer.self_attn_layer_norm.weight
        self.norm1_bias = mbart_layer.self_attn_layer_norm.bias
        self.norm2_eps = mbart_layer.final_layer_norm.eps
        self.norm2_weight = mbart_layer.final_layer_norm.weight
        self.norm2_bias = mbart_layer.final_layer_norm.bias
        self.num_heads = mbart_layer.self_attn.num_heads
        self.embed_dim = mbart_layer.self_attn.embed_dim
        self.is_last_layer = False
        self.norm_first = True
        self.original_layers_mapping = {'in_proj_weight': ['self_attn.q_proj.weight', 'self_attn.k_proj.weight', 'self_attn.v_proj.weight'], 'in_proj_bias': ['self_attn.q_proj.bias', 'self_attn.k_proj.bias', 'self_attn.v_proj.bias'], 'out_proj_weight': 'self_attn.out_proj.weight', 'out_proj_bias': 'self_attn.out_proj.bias', 'linear1_weight': 'fc1.weight', 'linear1_bias': 'fc1.bias', 'linear2_weight': 'fc2.weight', 'linear2_bias': 'fc2.bias', 'norm1_weight': 'self_attn_layer_norm.weight', 'norm1_bias': 'self_attn_layer_norm.bias', 'norm1_eps': 'self_attn_layer_norm.eps', 'norm2_weight': 'final_layer_norm.weight', 'norm2_bias': 'final_layer_norm.bias', 'norm2_eps': 'final_layer_norm.eps'}
        self.dropout = config.attention_dropout
        self.activation_dropout = config.activation_dropout
        self.attention_head_size = config.d_model // config.encoder_attention_heads
        self.act_fn_callable = ACT2FN[self.act_fn]
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
                if len(attention_mask.shape) == 4:
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
            residual = hidden_states
            hidden_states = F.layer_norm(hidden_states, normalized_shape=self.norm1_weight.shape, weight=self.norm1_weight, bias=self.norm1_bias)
            qkv = F.linear(hidden_states, weight=self.in_proj_weight, bias=self.in_proj_bias)
            qkv = qkv.view(qkv.size()[:-1] + (3, self.num_heads, self.attention_head_size)).permute(2, 0, 3, 1, 4)
            query, key, value = (qkv[0], qkv[1], qkv[2])
            if self.training:
                attention_mask = None
            attention_out = F.scaled_dot_product_attention(query, key, value, attn_mask=attention_mask, is_causal=False, dropout_p=self.dropout if self.training else 0.0)
            attention_out = attention_out.permute(0, 2, 1, 3).contiguous()
            new_attention_out_shape = attention_out.size()[:-2] + (self.num_heads * self.attention_head_size,)
            attention_out = attention_out.view(new_attention_out_shape)
            hidden_states = residual + F.dropout(F.linear(attention_out, self.out_proj_weight, self.out_proj_bias), p=self.dropout, training=self.training)
            residual = hidden_states
            hidden_states = F.layer_norm(hidden_states, normalized_shape=self.norm2_weight.shape, weight=self.norm2_weight, bias=self.norm2_bias)
            hidden_states = F.dropout(self.act_fn_callable(F.linear(hidden_states, self.linear1_weight, self.linear1_bias)), p=self.activation_dropout, training=self.training)
            hidden_states = residual + F.dropout(F.linear(hidden_states, self.linear2_weight, self.linear2_bias), p=self.dropout, training=self.training)
        return (hidden_states,)