import math
import warnings
from typing import Optional, Tuple, Union
import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, LayerNorm, MSELoss
from torch.nn import functional as F
from ...file_utils import add_code_sample_docstrings, add_start_docstrings, add_start_docstrings_to_model_forward
from ...modeling_attn_mask_utils import _prepare_4d_causal_attention_mask
from ...modeling_outputs import (
from ...modeling_utils import PreTrainedModel
from ...utils import logging
from .configuration_bloom import BloomConfig
@add_start_docstrings('The bare Bloom Model transformer outputting raw hidden-states without any specific head on top.', BLOOM_START_DOCSTRING)
class BloomModel(BloomPreTrainedModel):

    def __init__(self, config: BloomConfig):
        super().__init__(config)
        self.embed_dim = config.hidden_size
        self.num_heads = config.n_head
        self.word_embeddings = nn.Embedding(config.vocab_size, self.embed_dim)
        self.word_embeddings_layernorm = LayerNorm(self.embed_dim, eps=config.layer_norm_epsilon)
        self.h = nn.ModuleList([BloomBlock(config) for _ in range(config.num_hidden_layers)])
        self.ln_f = LayerNorm(self.embed_dim, eps=config.layer_norm_epsilon)
        self.gradient_checkpointing = False
        self.post_init()

    def build_alibi_tensor(self, attention_mask: torch.Tensor, num_heads: int, dtype: torch.dtype) -> torch.Tensor:
        return build_alibi_tensor(attention_mask, num_heads, dtype)

    def get_input_embeddings(self):
        return self.word_embeddings

    def set_input_embeddings(self, new_embeddings: torch.Tensor):
        self.word_embeddings = new_embeddings

    @add_start_docstrings_to_model_forward(BLOOM_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(checkpoint=_CHECKPOINT_FOR_DOC, output_type=BaseModelOutputWithPastAndCrossAttentions, config_class=_CONFIG_FOR_DOC)
    def forward(self, input_ids: Optional[torch.LongTensor]=None, past_key_values: Optional[Tuple[Tuple[torch.Tensor, torch.Tensor], ...]]=None, attention_mask: Optional[torch.Tensor]=None, head_mask: Optional[torch.LongTensor]=None, inputs_embeds: Optional[torch.LongTensor]=None, use_cache: Optional[bool]=None, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None, **deprecated_arguments) -> Union[Tuple[torch.Tensor, ...], BaseModelOutputWithPastAndCrossAttentions]:
        if deprecated_arguments.pop('position_ids', False) is not False:
            warnings.warn('`position_ids` have no functionality in BLOOM and will be removed in v5.0.0. You can safely ignore passing `position_ids`.', FutureWarning)
        if len(deprecated_arguments) > 0:
            raise ValueError(f'Got unexpected arguments: {deprecated_arguments}')
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
        head_mask = self.get_head_mask(head_mask, self.config.n_layer)
        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)
        hidden_states = self.word_embeddings_layernorm(inputs_embeds)
        presents = () if use_cache else None
        all_self_attentions = () if output_attentions else None
        all_hidden_states = () if output_hidden_states else None
        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once('`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`...')
                use_cache = False
        seq_length_with_past = seq_length
        past_key_values_length = 0
        if past_key_values[0] is not None:
            past_key_values_length = past_key_values[0][0].shape[2]
            seq_length_with_past = seq_length_with_past + past_key_values_length
        if attention_mask is None:
            attention_mask = torch.ones((batch_size, seq_length_with_past), device=hidden_states.device)
        else:
            attention_mask = attention_mask.to(hidden_states.device)
        alibi = self.build_alibi_tensor(attention_mask, self.num_heads, dtype=hidden_states.dtype)
        causal_mask = _prepare_4d_causal_attention_mask(attention_mask, input_shape=(batch_size, seq_length), inputs_embeds=inputs_embeds, past_key_values_length=past_key_values_length)
        causal_mask = causal_mask.bool()
        for i, (block, layer_past) in enumerate(zip(self.h, past_key_values)):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)
            if self.gradient_checkpointing and self.training:
                outputs = self._gradient_checkpointing_func(block.__call__, hidden_states, alibi, causal_mask, layer_past, head_mask[i], use_cache, output_attentions)
            else:
                outputs = block(hidden_states, layer_past=layer_past, attention_mask=causal_mask, head_mask=head_mask[i], use_cache=use_cache, output_attentions=output_attentions, alibi=alibi)
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