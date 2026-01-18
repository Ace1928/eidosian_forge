import copy
import inspect
import math
import random
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from ...activations import ACT2FN
from ...generation.configuration_utils import GenerationConfig
from ...generation.logits_process import ClassifierFreeGuidanceLogitsProcessor, LogitsProcessorList
from ...generation.stopping_criteria import StoppingCriteriaList
from ...modeling_attn_mask_utils import _prepare_4d_causal_attention_mask, _prepare_4d_causal_attention_mask_for_sdpa
from ...modeling_outputs import (
from ...modeling_utils import PreTrainedModel
from ...utils import (
from ..auto.configuration_auto import AutoConfig
from ..auto.modeling_auto import AutoModel, AutoModelForTextEncoding
from .configuration_musicgen_melody import MusicgenMelodyConfig, MusicgenMelodyDecoderConfig
from ..deprecated._archive_maps import MUSICGEN_MELODY_PRETRAINED_MODEL_ARCHIVE_LIST  # noqa: F401, E402
class MusicgenMelodyDecoder(MusicgenMelodyPreTrainedModel):
    """
    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`MusicgenMelodyDecoderLayer`]
    """

    def __init__(self, config: MusicgenMelodyDecoderConfig):
        super().__init__(config)
        self.dropout = config.dropout
        self.layerdrop = config.layerdrop
        self.max_target_positions = config.max_position_embeddings
        self.d_model = config.hidden_size
        self.num_codebooks = config.num_codebooks
        self.embed_scale = math.sqrt(config.hidden_size) if config.scale_embedding else 1.0
        embed_dim = config.vocab_size + 1
        self.embed_tokens = nn.ModuleList([nn.Embedding(embed_dim, config.hidden_size) for _ in range(config.num_codebooks)])
        self.embed_positions = MusicgenMelodySinusoidalPositionalEmbedding(config.max_position_embeddings, config.hidden_size)
        self.layers = nn.ModuleList([MusicgenMelodyDecoderLayer(config) for _ in range(config.num_hidden_layers)])
        self.layer_norm = nn.LayerNorm(config.hidden_size)
        self.attn_implementation = config._attn_implementation
        self.gradient_checkpointing = False
        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    @add_start_docstrings_to_model_forward(MUSICGEN_MELODY_DECODER_INPUTS_DOCSTRING)
    def forward(self, input_ids: torch.LongTensor=None, attention_mask: Optional[torch.Tensor]=None, encoder_hidden_states: Optional[torch.FloatTensor]=None, encoder_attention_mask: Optional[torch.LongTensor]=None, head_mask: Optional[torch.Tensor]=None, past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]]=None, inputs_embeds: Optional[torch.FloatTensor]=None, use_cache: Optional[bool]=None, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None) -> Union[Tuple, BaseModelOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError('You cannot specify both decoder_input_ids and decoder_inputs_embeds at the same time')
        elif input_ids is not None:
            input = input_ids.reshape(-1, self.num_codebooks, input_ids.shape[-1])
            bsz, num_codebooks, seq_len = input.shape
            input_shape = (bsz, seq_len)
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
            input = inputs_embeds[:, :, -1:]
        else:
            raise ValueError('You have to specify either decoder_input_ids or decoder_inputs_embeds')
        past_key_values_length = past_key_values[0][0].shape[2] if past_key_values is not None else 0
        if inputs_embeds is None:
            inputs_embeds = sum([self.embed_tokens[codebook](input[:, codebook]) for codebook in range(num_codebooks)])
        if encoder_hidden_states is not None:
            if encoder_attention_mask is not None and attention_mask is None:
                attention_mask = torch.ones(inputs_embeds.shape[:2], device=inputs_embeds.device)
            if attention_mask is not None:
                if encoder_attention_mask is None:
                    encoder_attention_mask = torch.ones(encoder_hidden_states.shape[:2], device=attention_mask.device)
                attention_mask = torch.cat([encoder_attention_mask, attention_mask], dim=1)
            inputs_embeds = torch.cat([encoder_hidden_states, inputs_embeds], dim=1)
        input_shape = inputs_embeds.size()[:-1]
        if self.attn_implementation == 'flash_attention_2':
            attention_mask = attention_mask if attention_mask is not None and 0 in attention_mask else None
        elif self.attn_implementation == 'sdpa' and (not output_attentions):
            attention_mask = _prepare_4d_causal_attention_mask_for_sdpa(attention_mask, input_shape, inputs_embeds, past_key_values_length)
        else:
            attention_mask = _prepare_4d_causal_attention_mask(attention_mask, input_shape, inputs_embeds, past_key_values_length)
        positions = self.embed_positions(inputs_embeds, past_key_values_length)
        hidden_states = inputs_embeds + positions.to(inputs_embeds.device)
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once('`use_cache=True` is incompatible with gradient checkpointing`. Setting `use_cache=False`...')
                use_cache = False
        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None
        next_decoder_cache = () if use_cache else None
        if head_mask is not None:
            if head_mask.size()[0] != len(self.layers):
                raise ValueError(f'The `head_mask` should be specified for {len(self.layers)} layers, but it is for {head_mask.size()[0]}.')
        for idx, decoder_layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
            dropout_probability = random.uniform(0, 1)
            if self.training and dropout_probability < self.layerdrop:
                continue
            past_key_value = past_key_values[idx] if past_key_values is not None else None
            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(decoder_layer.forward, hidden_states, attention_mask, head_mask[idx] if head_mask is not None else None, None, output_attentions, use_cache)
            else:
                layer_outputs = decoder_layer(hidden_states, attention_mask=attention_mask, layer_head_mask=head_mask[idx] if head_mask is not None else None, past_key_value=past_key_value, output_attentions=output_attentions, use_cache=use_cache)
            hidden_states = layer_outputs[0]
            if use_cache:
                next_decoder_cache += (layer_outputs[2 if output_attentions else 1],)
            if output_attentions:
                all_attentions += (layer_outputs[1],)
        hidden_states = self.layer_norm(hidden_states)
        if output_hidden_states:
            all_hidden_states += (hidden_states,)
        next_cache = next_decoder_cache if use_cache else None
        if not return_dict:
            return tuple((v for v in [hidden_states, next_cache, all_hidden_states, all_attentions] if v is not None))
        return BaseModelOutputWithPast(last_hidden_state=hidden_states, past_key_values=next_cache, hidden_states=all_hidden_states, attentions=all_attentions)