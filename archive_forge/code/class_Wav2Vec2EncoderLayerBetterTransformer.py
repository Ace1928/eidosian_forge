from typing import TYPE_CHECKING
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.activations import ACT2FN
from .base import BetterTransformerBaseLayer
class Wav2Vec2EncoderLayerBetterTransformer(BetterTransformerBaseLayer, nn.Module):

    def __init__(self, wav2vec2_layer, config):
        """
        A simple conversion of the Wav2Vec2EncoderLayer to its `BetterTransformer` implementation.

        Args:
            wav2vec2_layer (`torch.nn.Module`):
                The original `Wav2Vec2EncoderLayer` where the weights needs to be retrieved.
        """
        super().__init__(config)
        super(BetterTransformerBaseLayer, self).__init__()
        self.in_proj_weight = nn.Parameter(torch.cat([wav2vec2_layer.attention.q_proj.weight, wav2vec2_layer.attention.k_proj.weight, wav2vec2_layer.attention.v_proj.weight]))
        self.in_proj_bias = nn.Parameter(torch.cat([wav2vec2_layer.attention.q_proj.bias, wav2vec2_layer.attention.k_proj.bias, wav2vec2_layer.attention.v_proj.bias]))
        self.out_proj_weight = wav2vec2_layer.attention.out_proj.weight
        self.out_proj_bias = wav2vec2_layer.attention.out_proj.bias
        self.linear1_weight = wav2vec2_layer.feed_forward.intermediate_dense.weight
        self.linear1_bias = wav2vec2_layer.feed_forward.intermediate_dense.bias
        self.linear2_weight = wav2vec2_layer.feed_forward.output_dense.weight
        self.linear2_bias = wav2vec2_layer.feed_forward.output_dense.bias
        self.norm1_eps = wav2vec2_layer.layer_norm.eps
        self.norm1_weight = wav2vec2_layer.layer_norm.weight
        self.norm1_bias = wav2vec2_layer.layer_norm.bias
        self.norm2_eps = wav2vec2_layer.final_layer_norm.eps
        self.norm2_weight = wav2vec2_layer.final_layer_norm.weight
        self.norm2_bias = wav2vec2_layer.final_layer_norm.bias
        self.num_heads = wav2vec2_layer.attention.num_heads
        self.embed_dim = wav2vec2_layer.attention.embed_dim
        self.is_last_layer = False
        self.original_layers_mapping = {'in_proj_weight': ['attention.q_proj.weight', 'attention.k_proj.weight', 'attention.v_proj.weight'], 'in_proj_bias': ['attention.q_proj.bias', 'attention.k_proj.bias', 'attention.v_proj.bias'], 'out_proj_weight': 'attention.out_proj.weight', 'out_proj_bias': 'attention.out_proj.bias', 'linear1_weight': 'feed_forward.intermediate_dense.weight', 'linear1_bias': 'feed_forward.intermediate_dense.bias', 'linear2_weight': 'feed_forward.output_dense.weight', 'linear2_bias': 'feed_forward.output_dense.bias', 'norm1_weight': 'layer_norm.weight', 'norm1_bias': 'layer_norm.bias', 'norm1_eps': 'layer_norm.eps', 'norm2_weight': 'final_layer_norm.weight', 'norm2_bias': 'final_layer_norm.bias', 'norm2_eps': 'final_layer_norm.eps'}
        if config.do_stable_layer_norm:
            self.norm_first = True
        self.validate_bettertransformer()

    def forward(self, hidden_states, attention_mask, output_attentions: bool, **__):
        if output_attentions:
            raise ValueError('output_attentions=True can not be supported with BetterTransformer.')
        if not self.training and (not torch.is_autocast_enabled()) and (not torch.is_autocast_cpu_enabled()):
            if hidden_states.is_nested:
                attention_mask = None
            if attention_mask is not None:
                attention_mask = attention_mask.bool()
                if len(attention_mask.shape) == 4:
                    attention_mask = attention_mask.squeeze(1)[:, 0]
                attention_mask = torch.reshape(attention_mask, (attention_mask.shape[0], attention_mask.shape[-1]))
                hidden_states = torch._nested_tensor_from_mask(hidden_states, ~attention_mask)
                attention_mask = None
            hidden_states = torch._transformer_encoder_layer_fwd(hidden_states, self.embed_dim, self.num_heads, self.in_proj_weight, self.in_proj_bias, self.out_proj_weight, self.out_proj_bias, self.use_gelu, self.norm_first, self.norm1_eps, self.norm1_weight, self.norm1_bias, self.norm2_weight, self.norm2_bias, self.linear1_weight, self.linear1_bias, self.linear2_weight, self.linear2_bias, attention_mask)
            if hidden_states.is_nested and self.is_last_layer:
                hidden_states = hidden_states.to_padded_tensor(0.0)
        else:
            raise NotImplementedError('Training and Autocast are not implemented for BetterTransformer + Wav2Vec2. Please open an issue.')
        return (hidden_states,)