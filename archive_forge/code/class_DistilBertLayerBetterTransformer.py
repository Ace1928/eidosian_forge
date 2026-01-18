from typing import TYPE_CHECKING
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.activations import ACT2FN
from .base import BetterTransformerBaseLayer
class DistilBertLayerBetterTransformer(BetterTransformerBaseLayer, nn.Module):

    def __init__(self, bert_layer, config):
        """
        A simple conversion of the Distill-BERTLayer to its `BetterTransformer` implementation.

        Args:
            bert_layer (`torch.nn.Module`):
                The original Distill-BERT Layer where the weights needs to be retrieved.
        """
        super().__init__(config)
        super(BetterTransformerBaseLayer, self).__init__()
        self.in_proj_weight = nn.Parameter(torch.cat([bert_layer.attention.q_lin.weight, bert_layer.attention.k_lin.weight, bert_layer.attention.v_lin.weight]))
        self.in_proj_bias = nn.Parameter(torch.cat([bert_layer.attention.q_lin.bias, bert_layer.attention.k_lin.bias, bert_layer.attention.v_lin.bias]))
        self.out_proj_weight = bert_layer.attention.out_lin.weight
        self.out_proj_bias = bert_layer.attention.out_lin.bias
        self.linear1_weight = bert_layer.ffn.lin1.weight
        self.linear1_bias = bert_layer.ffn.lin1.bias
        self.linear2_weight = bert_layer.ffn.lin2.weight
        self.linear2_bias = bert_layer.ffn.lin2.bias
        self.norm1_eps = bert_layer.sa_layer_norm.eps
        self.norm1_weight = bert_layer.sa_layer_norm.weight
        self.norm1_bias = bert_layer.sa_layer_norm.bias
        self.norm2_eps = bert_layer.output_layer_norm.eps
        self.norm2_weight = bert_layer.output_layer_norm.weight
        self.norm2_bias = bert_layer.output_layer_norm.bias
        self.num_heads = bert_layer.attention.n_heads
        self.embed_dim = bert_layer.attention.dim
        self.is_last_layer = False
        self.original_layers_mapping = {'in_proj_weight': ['attention.q_lin.weight', 'attention.k_lin.weight', 'attention.v_lin.weight'], 'in_proj_bias': ['attention.q_lin.bias', 'attention.k_lin.bias', 'attention.v_lin.bias'], 'out_proj_weight': 'attention.out_lin.weight', 'out_proj_bias': 'attention.out_lin.bias', 'linear1_weight': 'ffn.lin1.weight', 'linear1_bias': 'ffn.lin1.bias', 'linear2_weight': 'ffn.lin2.weight', 'linear2_bias': 'ffn.lin2.bias', 'norm1_weight': 'sa_layer_norm.weight', 'norm1_bias': 'sa_layer_norm.bias', 'norm2_weight': 'output_layer_norm.weight', 'norm2_bias': 'output_layer_norm.bias'}
        self.attention_dropout = config.attention_dropout
        self.dropout = config.dropout
        self.attention_head_size = config.dim // config.n_heads
        self.act_fn_callable = ACT2FN[self.act_fn]
        self.validate_bettertransformer()

    def forward(self, hidden_states, attn_mask, output_attentions: bool, head_mask=None, *_):
        if output_attentions:
            raise ValueError('output_attentions=True can not be supported with BetterTransformer.')
        if not self.training and (not torch.is_autocast_enabled()) and (not torch.is_autocast_cpu_enabled()):
            if hidden_states.is_nested:
                attn_mask = None
            if attn_mask is not None:
                attn_mask = attn_mask.bool()
                attn_mask = torch.reshape(attn_mask, (attn_mask.shape[0], attn_mask.shape[-1]))
                seqlen = attn_mask.shape[1]
                lengths = torch.sum(~attn_mask, 1)
                if not all((l == seqlen for l in lengths)):
                    hidden_states = torch._nested_tensor_from_mask(hidden_states, attn_mask)
                attn_mask = None
            hidden_states = torch._transformer_encoder_layer_fwd(hidden_states, self.embed_dim, self.num_heads, self.in_proj_weight, self.in_proj_bias, self.out_proj_weight, self.out_proj_bias, self.use_gelu, self.norm_first, self.norm1_eps, self.norm1_weight, self.norm1_bias, self.norm2_weight, self.norm2_bias, self.linear1_weight, self.linear1_bias, self.linear2_weight, self.linear2_bias, attn_mask)
            if hidden_states.is_nested and self.is_last_layer:
                hidden_states = hidden_states.to_padded_tensor(0.0)
        else:
            qkv = F.linear(hidden_states, weight=self.in_proj_weight, bias=self.in_proj_bias)
            qkv = qkv.view(qkv.size()[:-1] + (3, self.num_heads, self.attention_head_size)).permute(2, 0, 3, 1, 4)
            query, key, value = (qkv[0], qkv[1], qkv[2])
            attn_mask = attn_mask.unsqueeze(1).unsqueeze(2).to(dtype=query.dtype)
            attn_mask = (1.0 - attn_mask) * torch.finfo(query.dtype).min
            if self.training:
                attn_mask = None
            attention_out = F.scaled_dot_product_attention(query, key, value, attn_mask=attn_mask, is_causal=False, dropout_p=self.attention_dropout if self.training else 0.0)
            attention_out = attention_out.permute(0, 2, 1, 3).contiguous()
            new_attention_out_shape = attention_out.size()[:-2] + (self.num_heads * self.attention_head_size,)
            attention_out = attention_out.view(new_attention_out_shape)
            attention_out = F.layer_norm(F.dropout(F.linear(attention_out, self.out_proj_weight, self.out_proj_bias), p=self.dropout, training=self.training) + hidden_states, normalized_shape=self.norm1_weight.shape, weight=self.norm1_weight, bias=self.norm1_bias)
            hidden_states = self.act_fn_callable(F.linear(attention_out, self.linear1_weight, self.linear1_bias))
            hidden_states = F.layer_norm(attention_out + F.dropout(F.linear(hidden_states, self.linear2_weight, self.linear2_bias), p=self.dropout, training=self.training), normalized_shape=self.norm2_weight.shape, weight=self.norm2_weight, bias=self.norm2_bias)
        return (hidden_states,)