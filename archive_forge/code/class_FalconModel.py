import math
import warnings
from typing import TYPE_CHECKING, Optional, Tuple, Union
import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, LayerNorm, MSELoss
from torch.nn import functional as F
from ...modeling_attn_mask_utils import (
from ...modeling_outputs import (
from ...modeling_utils import PreTrainedModel
from ...pytorch_utils import is_torch_greater_or_equal_than_2_0
from ...utils import (
from .configuration_falcon import FalconConfig
@add_start_docstrings('The bare Falcon Model transformer outputting raw hidden-states without any specific head on top.', FALCON_START_DOCSTRING)
class FalconModel(FalconPreTrainedModel):

    def __init__(self, config: FalconConfig):
        super().__init__(config)
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.use_alibi = config.alibi
        self.word_embeddings = nn.Embedding(config.vocab_size, self.embed_dim)
        self.h = nn.ModuleList([FalconDecoderLayer(config) for _ in range(config.num_hidden_layers)])
        self._use_flash_attention_2 = config._attn_implementation == 'flash_attention_2'
        self._use_sdpa = config._attn_implementation == 'sdpa'
        self.ln_f = LayerNorm(self.embed_dim, eps=config.layer_norm_epsilon)
        self.gradient_checkpointing = False
        self.post_init()

    def get_input_embeddings(self):
        return self.word_embeddings

    def set_input_embeddings(self, new_embeddings: torch.Tensor):
        self.word_embeddings = new_embeddings

    @add_start_docstrings_to_model_forward(FALCON_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(checkpoint=_CHECKPOINT_FOR_DOC, output_type=BaseModelOutputWithPastAndCrossAttentions, config_class=_CONFIG_FOR_DOC)
    def forward(self, input_ids: Optional[torch.LongTensor]=None, past_key_values: Optional[Tuple[Tuple[torch.Tensor, torch.Tensor], ...]]=None, attention_mask: Optional[torch.Tensor]=None, position_ids: Optional[torch.LongTensor]=None, head_mask: Optional[torch.LongTensor]=None, inputs_embeds: Optional[torch.LongTensor]=None, use_cache: Optional[bool]=None, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None) -> Union[Tuple[torch.Tensor, ...], BaseModelOutputWithPastAndCrossAttentions]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError('You cannot specify both input_ids and inputs_embeds at the same time')
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape
        elif inputs_embeds is not None:
            batch_size, seq_length, _ = inputs_embeds.shape
        else:
            raise ValueError('You have to specify either input_ids or inputs_embeds')
        if past_key_values is None:
            past_key_values = tuple([None] * len(self.h))
        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)
        hidden_states = inputs_embeds
        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning('`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`...')
                use_cache = False
        presents = () if use_cache else None
        all_self_attentions = () if output_attentions else None
        all_hidden_states = () if output_hidden_states else None
        past_key_values_length = 0
        if past_key_values[0] is not None:
            past_key_values_length = past_key_values[0][0].shape[-2]
        if self.use_alibi:
            mask = torch.ones((batch_size, seq_length + past_key_values_length), device=inputs_embeds.device, dtype=torch.long) if attention_mask is None else attention_mask
            alibi = build_alibi_tensor(mask, self.num_heads, dtype=hidden_states.dtype)
        else:
            alibi = None
            if position_ids is None:
                device = input_ids.device if input_ids is not None else inputs_embeds.device
                position_ids = torch.arange(past_key_values_length, seq_length + past_key_values_length, dtype=torch.long, device=device)
                position_ids = position_ids.unsqueeze(0)
        if self._use_flash_attention_2:
            attention_mask = attention_mask if attention_mask is not None and 0 in attention_mask else None
        elif self._use_sdpa and (not output_attentions):
            if alibi is None:
                attention_mask = _prepare_4d_causal_attention_mask_for_sdpa(attention_mask, (batch_size, seq_length), inputs_embeds, past_key_values_length)
            elif head_mask is None:
                alibi = alibi.reshape(batch_size, -1, *alibi.shape[1:])
                attention_mask_2d = attention_mask
                attention_mask = _prepare_4d_causal_attention_mask(attention_mask, (batch_size, seq_length), inputs_embeds, past_key_values_length)
                if attention_mask_2d is None:
                    attention_mask = alibi / math.sqrt(self.config.hidden_size // self.num_heads)
                else:
                    attention_mask = torch.masked_fill(alibi / math.sqrt(self.config.hidden_size // self.num_heads), attention_mask < -1, torch.finfo(alibi.dtype).min)
                    if seq_length > 1:
                        attention_mask = AttentionMaskConverter._unmask_unattended(attention_mask, attention_mask_2d, unmasked_value=0.0)
            else:
                attention_mask = _prepare_4d_causal_attention_mask(attention_mask, (batch_size, seq_length), inputs_embeds, past_key_values_length)
        else:
            attention_mask = _prepare_4d_causal_attention_mask(attention_mask, (batch_size, seq_length), inputs_embeds, past_key_values_length)
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)
        for i, (block, layer_past) in enumerate(zip(self.h, past_key_values)):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)
            if self.gradient_checkpointing and self.training:
                outputs = self._gradient_checkpointing_func(block.__call__, hidden_states, alibi, attention_mask, position_ids, head_mask[i], layer_past, use_cache, output_attentions)
            else:
                outputs = block(hidden_states, layer_past=layer_past, attention_mask=attention_mask, position_ids=position_ids, head_mask=head_mask[i], use_cache=use_cache, output_attentions=output_attentions, alibi=alibi)
            hidden_states = outputs[0]
            if use_cache is True:
                presents = presents + (outputs[1],)
            if output_attentions:
                all_self_attentions = all_self_attentions + (outputs[2 if use_cache else 1],)
        hidden_states = self.ln_f(hidden_states)
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)
        if not return_dict:
            return tuple((v for v in [hidden_states, presents, all_hidden_states, all_self_attentions] if v is not None))
        return BaseModelOutputWithPastAndCrossAttentions(last_hidden_state=hidden_states, past_key_values=presents, hidden_states=all_hidden_states, attentions=all_self_attentions)