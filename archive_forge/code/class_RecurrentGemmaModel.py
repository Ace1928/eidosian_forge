import math
from typing import Dict, Optional, Tuple, Union
import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import CrossEntropyLoss
from ...activations import ACT2FN
from ...modeling_attn_mask_utils import AttentionMaskConverter
from ...modeling_outputs import BaseModelOutputWithNoAttention, CausalLMOutput
from ...modeling_utils import PreTrainedModel
from ...pytorch_utils import ALL_LAYERNORM_LAYERS
from ...utils import (
from .configuration_recurrent_gemma import RecurrentGemmaConfig
@add_start_docstrings('The bare RecurrentGemma Model outputting raw hidden-states without any specific head on top.', RECURRENTGEMMA_START_DOCSTRING)
class RecurrentGemmaModel(RecurrentGemmaPreTrainedModel):
    """
    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`RecurrentGemmaDecoderLayer`]

    Args:
        config: RecurrentGemmaConfig
    """

    def __init__(self, config: RecurrentGemmaConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList([RecurrentGemmaDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)])
        self.final_norm = RecurrentGemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.gradient_checkpointing = False
        self.register_buffer('normalizer', torch.tensor(self.config.hidden_size ** 0.5, dtype=torch.bfloat16), persistent=False)
        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    @add_start_docstrings_to_model_forward(RECURRENTGEMMA_INPUTS_DOCSTRING)
    def forward(self, input_ids: torch.LongTensor=None, position_ids: Optional[torch.LongTensor]=None, attention_mask: Optional[torch.Tensor]=None, cache_position: Optional[torch.LongTensor]=None, inputs_embeds: Optional[torch.FloatTensor]=None, use_cache: Optional[bool]=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None, **kwargs) -> Union[Tuple, BaseModelOutputWithNoAttention]:
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError('You cannot specify both input_ids and inputs_embeds at the same time, and must specify either one')
        if self.gradient_checkpointing and self.training and use_cache:
            logger.warning_once('`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`.')
            use_cache = False
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        hidden_states = inputs_embeds
        if use_cache and inputs_embeds.shape[1] != 1:
            self._setup_cache(self.config, hidden_states.shape[0], hidden_states.device, hidden_states.dtype)
        if cache_position is None:
            cache_position = torch.arange(hidden_states.shape[1], device=hidden_states.device)
        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)
        causal_mask = self._update_causal_mask(attention_mask, inputs_embeds, cache_position)
        hidden_states = hidden_states * self.normalizer.type(hidden_states.dtype)
        all_hidden_states = () if output_hidden_states else None
        for i, residual_block in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
            if self.gradient_checkpointing and self.training:
                hidden_states = self._gradient_checkpointing_func(residual_block.__call__, hidden_states, position_ids, causal_mask, cache_position, use_cache)
            else:
                hidden_states = residual_block(hidden_states, position_ids, causal_mask, cache_position, use_cache)
        hidden_states = self.final_norm(hidden_states)
        if output_hidden_states:
            all_hidden_states += (hidden_states,)
        if not return_dict:
            return tuple((v for v in [hidden_states, all_hidden_states] if v is not None))
        return BaseModelOutputWithNoAttention(last_hidden_state=hidden_states, hidden_states=all_hidden_states)

    def _update_causal_mask(self, attention_mask, input_tensor, cache_position):
        dtype, device = (input_tensor.dtype, input_tensor.device)
        min_dtype = torch.finfo(dtype).min
        sequence_length = input_tensor.shape[1]
        target_length = max(self.config.attention_window_size, sequence_length)
        diagonal = torch.full((sequence_length, target_length), fill_value=min_dtype, dtype=dtype, device=device)
        causal_mask = diagonal
        if sequence_length != 1:
            causal_mask = torch.triu(diagonal, diagonal=-1)
        causal_mask *= torch.arange(target_length, device=device) > cache_position.reshape(-1, 1)
        causal_mask = causal_mask[None, None, :, :].expand(input_tensor.shape[0], 1, -1, -1)
        if attention_mask is not None:
            causal_mask = causal_mask.clone()
            if attention_mask.dim() == 2:
                mask_length = attention_mask.shape[-1]
                padding_mask = causal_mask[..., :mask_length].eq(0.0) * attention_mask[:, None, None, :].eq(0.0)
                causal_mask[..., :mask_length] = causal_mask[..., :mask_length].masked_fill(padding_mask, min_dtype)
        if attention_mask is not None and attention_mask.device.type == 'cuda':
            causal_mask = AttentionMaskConverter._unmask_unattended(causal_mask, min_dtype)
        return causal_mask