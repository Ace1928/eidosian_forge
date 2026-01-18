from typing import TYPE_CHECKING
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.activations import ACT2FN
from .base import BetterTransformerBaseLayer
class CLIPLayerBetterTransformer(BetterTransformerBaseLayer, nn.Module):

    def __init__(self, layer, config):
        """
        A simple conversion of the CLIPEncoderLayer to its `BetterTransformer` implementation.

        **The implementation is valid only for the vision model, that does not use `causal_attention_mask`.**

        Args:
            layer (`torch.nn.Module`):
                The original `CLIPEncoderLayer` where the weights needs to be retrieved.
        """
        super().__init__(config)
        super(BetterTransformerBaseLayer, self).__init__()
        self.in_proj_weight = nn.Parameter(torch.cat([layer.self_attn.q_proj.weight, layer.self_attn.k_proj.weight, layer.self_attn.v_proj.weight]))
        self.in_proj_bias = nn.Parameter(torch.cat([layer.self_attn.q_proj.bias, layer.self_attn.k_proj.bias, layer.self_attn.v_proj.bias]))
        self.out_proj_weight = layer.self_attn.out_proj.weight
        self.out_proj_bias = layer.self_attn.out_proj.bias
        self.linear1_weight = layer.mlp.fc1.weight
        self.linear1_bias = layer.mlp.fc1.bias
        self.linear2_weight = layer.mlp.fc2.weight
        self.linear2_bias = layer.mlp.fc2.bias
        self.norm1_eps = layer.layer_norm1.eps
        self.norm1_weight = layer.layer_norm1.weight
        self.norm1_bias = layer.layer_norm1.bias
        self.norm2_eps = layer.layer_norm2.eps
        self.norm2_weight = layer.layer_norm2.weight
        self.norm2_bias = layer.layer_norm2.bias
        self.num_heads = layer.self_attn.num_heads
        self.embed_dim = layer.self_attn.embed_dim
        self.is_last_layer = False
        self.norm_first = True
        self.original_layers_mapping = {'in_proj_weight': ['self_attn.q_proj.weight', 'self_attn.k_proj.weight', 'self_attn.v_proj.weight'], 'in_proj_bias': ['self_attn.q_proj.bias', 'self_attn.k_proj.bias', 'self_attn.v_proj.bias'], 'out_proj_weight': 'self_attn.out_proj.weight', 'out_proj_bias': 'self_attn.out_proj.bias', 'linear1_weight': 'mlp.fc1.weight', 'linear1_bias': 'mlp.fc1.bias', 'linear2_weight': 'mlp.fc2.weight', 'linear2_bias': 'mlp.fc2.bias', 'norm1_eps': 'layer_norm1.eps', 'norm1_weight': 'layer_norm1.weight', 'norm1_bias': 'layer_norm1.bias', 'norm2_eps': 'layer_norm2.eps', 'norm2_weight': 'layer_norm2.weight', 'norm2_bias': 'layer_norm2.bias'}
        self.validate_bettertransformer()

    def forward(self, hidden_states, attention_mask, causal_attention_mask, output_attentions: bool, *_, **__):
        if output_attentions:
            raise ValueError('output_attentions=True can not be supported with BetterTransformer.')
        if not self.training and (not torch.is_autocast_enabled()) and (not torch.is_autocast_cpu_enabled()):
            if attention_mask is not None or causal_attention_mask is not None:
                raise ValueError('Please do not use attention masks when using `BetterTransformer` converted vision models')
            hidden_states = torch._transformer_encoder_layer_fwd(hidden_states, self.embed_dim, self.num_heads, self.in_proj_weight, self.in_proj_bias, self.out_proj_weight, self.out_proj_bias, self.use_gelu, self.norm_first, self.norm1_eps, self.norm1_weight, self.norm1_bias, self.norm2_weight, self.norm2_bias, self.linear1_weight, self.linear1_bias, self.linear2_weight, self.linear2_bias, attention_mask)
        else:
            raise NotImplementedError('Training and Autocast are not implemented for BetterTransformer + CLIP. Please open an issue.')
        return (hidden_states,)

    def _get_activation_function(self, config: 'PretrainedConfig'):
        if hasattr(config, 'vision_config') and hasattr(config, 'text_config'):
            assert config.vision_config.hidden_act == config.text_config.hidden_act
            return config.vision_config.hidden_act
        else:
            return config.hidden_act