from typing import Optional, Tuple, Union
import torch
import torch.utils.checkpoint
from torch import Tensor, nn
from torch.nn import CrossEntropyLoss
from ...activations import ACT2FN
from ...file_utils import add_start_docstrings, add_start_docstrings_to_model_forward, replace_return_docstrings
from ...modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from ...modeling_utils import PreTrainedModel
from ...utils import logging
from .configuration_gpt_neox_japanese import GPTNeoXJapaneseConfig
class GPTNeoXJapaneseLayer(nn.Module):

    def __init__(self, config, layer_number):
        super().__init__()
        self.layer_number = layer_number
        self.input_layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.post_attention_layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.attention = GPTNeoXJapaneseAttention(config=config, use_bias=layer_number == config.num_hidden_layers - 1)
        self.mlp = GPTNeoXJapaneseMLP(config)
        self.hidden_dropout = config.hidden_dropout

    def forward(self, hidden_states, attention_mask=None, head_mask=None, use_cache=False, layer_past=None, output_attentions=False):
        residual = hidden_states
        ln_out = self.input_layernorm(hidden_states)
        attention_layer_outputs, attn_bias = self.attention(ln_out, attention_mask=attention_mask, layer_past=layer_past, head_mask=head_mask, use_cache=use_cache, output_attentions=output_attentions)
        attn_output = attention_layer_outputs[0]
        outputs = attention_layer_outputs[1:]
        attn_output = bias_dropout_add(attn_output, bias=attn_bias.expand_as(residual) if attn_bias is not None else attn_bias, residual=residual, prob=self.hidden_dropout, training=self.training)
        mlp_output = self.mlp(self.post_attention_layernorm(attn_output))
        attn_output = bias_dropout_add(mlp_output, bias=None, residual=attn_output, prob=self.hidden_dropout, training=self.training)
        if use_cache:
            outputs = (attn_output,) + outputs
        else:
            outputs = (attn_output,) + outputs[1:]
        return outputs