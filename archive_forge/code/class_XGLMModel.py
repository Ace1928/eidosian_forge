import math
from typing import List, Optional, Tuple, Union
import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import CrossEntropyLoss
from ...activations import ACT2FN
from ...modeling_attn_mask_utils import _prepare_4d_attention_mask, _prepare_4d_causal_attention_mask
from ...modeling_outputs import BaseModelOutputWithPastAndCrossAttentions, CausalLMOutputWithCrossAttentions
from ...modeling_utils import PreTrainedModel
from ...utils import add_code_sample_docstrings, add_start_docstrings, add_start_docstrings_to_model_forward, logging
from .configuration_xglm import XGLMConfig
@add_start_docstrings('The bare XGLM Model transformer outputting raw hidden-states without any specific head on top.', XGLM_START_DOCSTRING)
class XGLMModel(XGLMPreTrainedModel):
    """
    Transformer decoder consisting of *config.num_layers* layers. Each layer is a [`XGLMDecoderLayer`]

    Args:
        config: XGLMConfig
        embed_tokens (nn.Embedding): output embedding
    """

    def __init__(self, config: XGLMConfig, embed_tokens: Optional[nn.Embedding]=None):
        super().__init__(config)
        self.dropout = config.dropout
        self.layerdrop = config.layerdrop
        self.padding_idx = config.pad_token_id
        self.max_target_positions = config.max_position_embeddings
        self.embed_scale = math.sqrt(config.d_model) if config.scale_embedding else 1.0
        if embed_tokens is not None:
            self.embed_tokens = embed_tokens
        else:
            self.embed_tokens = nn.Embedding(config.vocab_size, config.d_model, self.padding_idx)
        self.embed_positions = XGLMSinusoidalPositionalEmbedding(config.max_position_embeddings, config.d_model, config.pad_token_id)
        self.layers = nn.ModuleList([XGLMDecoderLayer(config) for _ in range(config.num_layers)])
        self.layer_norm = nn.LayerNorm(config.d_model)
        self.gradient_checkpointing = False
        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    @add_start_docstrings_to_model_forward(XGLM_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(checkpoint=_CHECKPOINT_FOR_DOC, output_type=BaseModelOutputWithPastAndCrossAttentions, config_class=_CONFIG_FOR_DOC)
    def forward(self, input_ids: Optional[torch.Tensor]=None, attention_mask: Optional[torch.Tensor]=None, position_ids: Optional[torch.Tensor]=None, encoder_hidden_states: Optional[torch.Tensor]=None, encoder_attention_mask: Optional[torch.Tensor]=None, head_mask: Optional[torch.Tensor]=None, cross_attn_head_mask: Optional[torch.Tensor]=None, past_key_values: Optional[List[torch.FloatTensor]]=None, inputs_embeds: Optional[torch.Tensor]=None, use_cache: Optional[bool]=None, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None) -> Union[Tuple[torch.Tensor], BaseModelOutputWithPastAndCrossAttentions]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError('You cannot specify both input_ids and inputs_embeds at the same time')
        elif input_ids is not None:
            self.warn_if_padding_and_no_attention_mask(input_ids, attention_mask)
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError('You have to specify either input_ids or inputs_embeds')
        past_key_values_length = past_key_values[0][0].shape[2] if past_key_values is not None else 0
        if position_ids is None:
            position_ids = torch.arange(past_key_values_length, input_shape[-1] + past_key_values_length, dtype=torch.long, device=input_ids.device if input_ids is not None else inputs_embeds.device)
            position_ids = position_ids.unsqueeze(0)
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids) * self.embed_scale
        attention_mask = _prepare_4d_causal_attention_mask(attention_mask, input_shape, inputs_embeds, past_key_values_length)
        if encoder_hidden_states is not None and encoder_attention_mask is not None:
            encoder_attention_mask = _prepare_4d_attention_mask(encoder_attention_mask, inputs_embeds.dtype, tgt_len=input_shape[-1])
        hidden_states = inputs_embeds + self.embed_positions(position_ids, past_key_values_length)
        hidden_states = nn.functional.dropout(hidden_states, p=float(self.dropout), training=self.training)
        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once('`use_cache = True` is incompatible with gradient checkpointing`. Setting `use_cache = False`...')
                use_cache = False
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        all_cross_attentions = () if output_attentions and encoder_hidden_states is not None else None
        next_decoder_cache = () if use_cache else None
        for attn_mask, mask_name in zip([head_mask, cross_attn_head_mask], ['head_mask', 'cross_attn_head_mask']):
            if attn_mask is not None:
                if attn_mask.size()[0] != len(self.layers):
                    raise ValueError(f'The `{mask_name}` should be specified for {len(self.layers)} layers, but it is for {head_mask.size()[0]}.')
        for idx, decoder_layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
            if self.training:
                dropout_probability = torch.rand([])
                if dropout_probability < self.layerdrop:
                    continue
            past_key_value = past_key_values[idx] if past_key_values is not None else None
            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(decoder_layer.__call__, hidden_states, attention_mask, encoder_hidden_states, encoder_attention_mask, head_mask[idx] if head_mask is not None else None, cross_attn_head_mask[idx] if cross_attn_head_mask is not None else None, None, output_attentions, use_cache)
            else:
                layer_outputs = decoder_layer(hidden_states, attention_mask=attention_mask, encoder_hidden_states=encoder_hidden_states, encoder_attention_mask=encoder_attention_mask, layer_head_mask=head_mask[idx] if head_mask is not None else None, cross_attn_layer_head_mask=cross_attn_head_mask[idx] if cross_attn_head_mask is not None else None, past_key_value=past_key_value, output_attentions=output_attentions, use_cache=use_cache)
            hidden_states = layer_outputs[0]
            if use_cache:
                next_decoder_cache += (layer_outputs[3 if output_attentions else 1],)
            if output_attentions:
                all_self_attns += (layer_outputs[1],)
                if encoder_hidden_states is not None:
                    all_cross_attentions += (layer_outputs[2],)
        hidden_states = self.layer_norm(hidden_states)
        if output_hidden_states:
            all_hidden_states += (hidden_states,)
        next_cache = next_decoder_cache if use_cache else None
        if not return_dict:
            return tuple((v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns, all_cross_attentions] if v is not None))
        return BaseModelOutputWithPastAndCrossAttentions(last_hidden_state=hidden_states, past_key_values=next_cache, hidden_states=all_hidden_states, attentions=all_self_attns, cross_attentions=all_cross_attentions)