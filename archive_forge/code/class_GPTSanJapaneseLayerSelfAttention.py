import copy
from typing import List, Optional, Tuple, Union
import torch
import torch.nn as nn
from ...activations import ACT2FN
from ...modeling_outputs import MoECausalLMOutputWithPast, MoEModelOutputWithPastAndCrossAttentions
from ...modeling_utils import PreTrainedModel
from ...utils import (
from .configuration_gptsan_japanese import GPTSanJapaneseConfig
class GPTSanJapaneseLayerSelfAttention(nn.Module):
    """
    Self Attention and Normalization Unit
    """

    def __init__(self, config, has_relative_attention_bias=False):
        super().__init__()
        self.self_attn = GPTSanJapaneseAttention(embed_dim=config.d_model, num_heads=config.num_heads, is_decoder=True, bias=has_relative_attention_bias)
        self.norm = nn.LayerNorm(config.d_model, eps=config.layer_norm_epsilon)

    def forward(self, hidden_states: Optional[Tuple[torch.FloatTensor]], past_key_value: Optional[Tuple[torch.Tensor]]=None, attention_mask: Optional[torch.FloatTensor]=None, head_mask: Optional[torch.FloatTensor]=None, use_cache: Optional[bool]=False, output_attentions: Optional[bool]=False) -> Tuple[Union[torch.Tensor, Tuple[torch.Tensor]], ...]:
        """
        Self-attention and normalize block.

        Args:
            hidden_states  (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
                Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention
                if the model is configured as a decoder.
            past_key_values (`tuple(tuple(torch.FloatTensor))` of length `config.n_layers` with each tuple having 4 tensors of shape `(batch_size, num_heads, sequence_length - 1, embed_size_per_head)`):
                Contains precomputed key and value hidden states of the attention blocks. Can be used to speed up
                decoding. If `past_key_values` are used, the user can optionally input only the last
                `decoder_input_ids` (those that don't have their past key value states given to this model) of shape
                `(batch_size, 1)` instead of all `decoder_input_ids` of shape `(batch_size, sequence_length)`.
            attention_mask (`torch.FloatTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Mask to avoid performing attention on the padding token indices of the encoder input. This mask is used
                in the cross-attention if the model is configured as a decoder. Mask values selected in `[0, 1]`:

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.

            head_mask (`numpy.ndarray` of shape `({0})`, `optional):
                Mask to nullify selected heads of the attention modules. Mask values selected in `[0, 1]`:

                - 1 indicates the head is **not masked**,
                - 0 indicates the head is **masked**.

            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
        Returns:
            Tuple[torch.Tensor[num_groups, tokens_per_group, hidden_dim],...]
        """
        self_attn_past_key_value = past_key_value[:2] if past_key_value is not None else None
        atten_out = self.self_attn(hidden_states=hidden_states, past_key_value=self_attn_past_key_value, attention_mask=(1 - attention_mask) * torch.finfo(hidden_states.dtype).min, layer_head_mask=head_mask, output_attentions=output_attentions)
        if output_attentions:
            attn_weights = (atten_out[1],)
        else:
            attn_weights = ()
        attention_output = atten_out[0]
        hidden = hidden_states + self.norm(attention_output)
        if use_cache:
            outputs = (hidden, atten_out[2])
        else:
            outputs = (hidden,)
        return outputs + attn_weights