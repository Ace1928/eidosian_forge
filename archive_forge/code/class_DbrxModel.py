import math
from typing import Any, Dict, Optional, Tuple, Union
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn
from ...activations import ACT2FN
from ...cache_utils import Cache, DynamicCache, StaticCache
from ...modeling_attn_mask_utils import AttentionMaskConverter
from ...modeling_outputs import MoeCausalLMOutputWithPast, MoeModelOutputWithPast
from ...modeling_utils import PreTrainedModel
from ...utils import (
from .configuration_dbrx import DbrxConfig
@add_start_docstrings('The bare DBRX Model outputting raw hidden-states without any specific head on top.', DBRX_START_DOCSTRING)
class DbrxModel(DbrxPreTrainedModel):
    """Transformer decoder consisting of *config.num_hidden_layers*. Each layer is a [`DbrxBlock`] layer.

    Args:
        config ([`DbrxConfig`]): Model configuration class with all parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
    """

    def __init__(self, config: DbrxConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.emb_pdrop = config.emb_pdrop
        self.wte = nn.Embedding(config.vocab_size, config.d_model, self.padding_idx)
        self.blocks = nn.ModuleList([DbrxBlock(config, block_idx) for block_idx in range(config.n_layers)])
        self.norm_f = nn.LayerNorm(config.d_model, bias=False)
        self.gradient_checkpointing = False
        self.post_init()

    def get_input_embeddings(self) -> nn.Embedding:
        return self.wte

    def set_input_embeddings(self, value: nn.Embedding):
        self.wte = value

    @add_start_docstrings_to_model_forward(DBRX_INPUTS_DOCSTRING)
    def forward(self, input_ids: Optional[torch.LongTensor]=None, attention_mask: Optional[torch.Tensor]=None, position_ids: Optional[torch.LongTensor]=None, past_key_values: Optional[Cache]=None, inputs_embeds: Optional[torch.Tensor]=None, use_cache: Optional[bool]=None, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, output_router_logits: Optional[bool]=None, return_dict: Optional[bool]=None, cache_position: Optional[torch.LongTensor]=None) -> Union[Tuple, MoeModelOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        output_router_logits = output_router_logits if output_router_logits is not None else self.config.output_router_logits
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError('You cannot specify both input_ids and inputs_embeds at the same time, and must specify either one')
        if self.gradient_checkpointing and self.training and use_cache:
            logger.warning_once('`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`.')
            use_cache = False
        if inputs_embeds is None:
            inputs_embeds = self.wte(input_ids)
        inputs_embeds = nn.functional.dropout(inputs_embeds, p=self.emb_pdrop, training=self.training)
        past_seen_tokens = 0
        if use_cache:
            if not isinstance(past_key_values, StaticCache):
                past_key_values = DynamicCache.from_legacy_cache(past_key_values)
                past_seen_tokens = past_key_values.get_seq_length()
        if cache_position is None:
            if isinstance(past_key_values, StaticCache):
                raise ValueError('cache_position is a required argument when using StaticCache.')
            cache_position = torch.arange(past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device)
        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)
        causal_mask = self._update_causal_mask(attention_mask, inputs_embeds, cache_position)
        hidden_states = inputs_embeds
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        all_router_logits = () if output_router_logits else None
        next_decoder_cache = None
        for block in self.blocks:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
            if self.gradient_checkpointing and self.training:
                block_outputs = self._gradient_checkpointing_func(block.__call__, hidden_states, causal_mask, position_ids, past_key_values, output_attentions, output_router_logits, use_cache, cache_position)
            else:
                block_outputs = block(hidden_states, attention_mask=causal_mask, position_ids=position_ids, past_key_value=past_key_values, output_attentions=output_attentions, output_router_logits=output_router_logits, use_cache=use_cache, cache_position=cache_position)
            hidden_states = block_outputs[0]
            if use_cache:
                next_decoder_cache = block_outputs[2 if output_attentions else 1]
            if output_attentions:
                all_self_attns += (block_outputs[1],)
            if output_router_logits:
                all_router_logits += (block_outputs[-1],)
        hidden_states = self.norm_f(hidden_states)
        if output_hidden_states:
            all_hidden_states += (hidden_states,)
        next_cache = None
        if use_cache:
            next_cache = next_decoder_cache.to_legacy_cache() if isinstance(next_decoder_cache, Cache) else next_decoder_cache
        if not return_dict:
            return tuple((v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns, all_router_logits] if v is not None))
        return MoeModelOutputWithPast(last_hidden_state=hidden_states, past_key_values=next_cache, hidden_states=all_hidden_states, attentions=all_self_attns, router_logits=all_router_logits)

    def _update_causal_mask(self, attention_mask: Optional[torch.Tensor], input_tensor: torch.Tensor, cache_position: torch.Tensor) -> Optional[torch.Tensor]:
        if self.config._attn_implementation == 'flash_attention_2':
            if attention_mask is not None and 0.0 in attention_mask:
                return attention_mask
            return None
        dtype, device = (input_tensor.dtype, input_tensor.device)
        min_dtype = torch.finfo(dtype).min
        sequence_length = input_tensor.shape[1]
        if hasattr(self.blocks[0].norm_attn_norm.attn, 'past_key_value'):
            target_length = self.config.max_position_embeddings
        else:
            target_length = attention_mask.shape[-1] if isinstance(attention_mask, torch.Tensor) else cache_position[-1] + 1
        target_length = int(target_length)
        causal_mask = torch.full((sequence_length, target_length), fill_value=min_dtype, dtype=dtype, device=device)
        if sequence_length != 1:
            causal_mask = torch.triu(causal_mask, diagonal=1)
        causal_mask *= torch.arange(target_length, device=device) > cache_position.reshape(-1, 1)
        causal_mask = causal_mask[None, None, :, :].expand(input_tensor.shape[0], 1, -1, -1)
        if attention_mask is not None:
            causal_mask = causal_mask.clone()
            if attention_mask.dim() == 2:
                mask_length = attention_mask.shape[-1]
                padding_mask = causal_mask[..., :mask_length].eq(0.0) * attention_mask[:, None, None, :].eq(0.0)
                causal_mask[..., :mask_length] = causal_mask[..., :mask_length].masked_fill(padding_mask, min_dtype)
            elif attention_mask.dim() == 4:
                if attention_mask.shape[-2] < cache_position[0] + sequence_length:
                    offset = cache_position[0]
                else:
                    offset = 0
                mask_shape = attention_mask.shape
                mask_slice = attention_mask.eq(0.0).to(dtype=dtype) * min_dtype
                causal_mask[:mask_shape[0], :mask_shape[1], offset:mask_shape[2] + offset, :mask_shape[3]] = mask_slice
        if self.config._attn_implementation == 'sdpa' and attention_mask is not None and (attention_mask.device.type == 'cuda'):
            is_tracing = torch.jit.is_tracing() or isinstance(input_tensor, torch.fx.Proxy) or (hasattr(torch, '_dynamo') and torch._dynamo.is_compiling())
            if not is_tracing and torch.any(attention_mask != 1):
                causal_mask = AttentionMaskConverter._unmask_unattended(causal_mask, min_dtype)
        return causal_mask