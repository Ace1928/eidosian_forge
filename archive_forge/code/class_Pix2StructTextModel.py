import math
from typing import Dict, List, Optional, Tuple, Union
import torch
import torch.utils.checkpoint
from torch import nn
from ...activations import ACT2FN
from ...modeling_outputs import (
from ...modeling_utils import PreTrainedModel
from ...pytorch_utils import ALL_LAYERNORM_LAYERS
from ...utils import (
from .configuration_pix2struct import Pix2StructConfig, Pix2StructTextConfig, Pix2StructVisionConfig
@add_start_docstrings('The standalone text decoder of Pix2Struct', PIX2STRUCT_START_DOCSTRING)
class Pix2StructTextModel(Pix2StructPreTrainedModel):
    config_class = Pix2StructTextConfig
    _no_split_modules = ['Pix2StructTextBlock']
    _tied_weights_keys = ['lm_head.weight']
    supports_gradient_checkpointing = True

    def __init__(self, config):
        super().__init__(config)
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layer = nn.ModuleList([Pix2StructTextBlock(config, has_relative_attention_bias=bool(i == 0)) for i in range(config.num_layers)])
        self.final_layer_norm = Pix2StructLayerNorm(config.hidden_size, eps=config.layer_norm_epsilon)
        self.dropout = nn.Dropout(config.dropout_rate)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()
        self.gradient_checkpointing = False

    def _reorder_cache(self, past_key_values, beam_idx):
        if past_key_values is None:
            logger.warning('You might want to consider setting `use_cache=True` to speed up decoding')
            return past_key_values
        reordered_decoder_past = ()
        for layer_past_states in past_key_values:
            reordered_layer_past_states = ()
            for layer_past_state in layer_past_states:
                reordered_layer_past_states = reordered_layer_past_states + (layer_past_state.index_select(0, beam_idx.to(layer_past_state.device)),)
            if reordered_layer_past_states[0].shape != layer_past_states[0].shape:
                raise ValueError(f'reordered_layer_past_states[0] shape {reordered_layer_past_states[0].shape} and layer_past_states[0] shape {layer_past_states[0].shape} mismatched')
            if len(reordered_layer_past_states) != len(layer_past_states):
                raise ValueError(f'length of reordered_layer_past_states {len(reordered_layer_past_states)} and length of layer_past_states {len(layer_past_states)} mismatched')
            reordered_decoder_past = reordered_decoder_past + (reordered_layer_past_states,)
        return reordered_decoder_past

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, new_embeddings):
        self.embed_tokens = new_embeddings

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    @add_start_docstrings_to_model_forward(PIX2STRUCT_TEXT_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=CausalLMOutputWithCrossAttentions, config_class=_CONFIG_FOR_DOC)
    def forward(self, input_ids: Optional[torch.LongTensor]=None, attention_mask: Optional[torch.FloatTensor]=None, encoder_hidden_states: Optional[torch.FloatTensor]=None, encoder_attention_mask: Optional[torch.FloatTensor]=None, inputs_embeds: Optional[torch.LongTensor]=None, head_mask: Optional[torch.FloatTensor]=None, cross_attn_head_mask: Optional[torch.Tensor]=None, past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]]=None, use_cache: Optional[bool]=None, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, labels: Optional[torch.LongTensor]=None, return_dict: Optional[bool]=None, **kwargs) -> Union[Tuple[torch.FloatTensor, ...], CausalLMOutputWithCrossAttentions]:
        """
        Returns:

        Example:

        ```python
        >>> from transformers import AutoProcessor, Pix2StructTextModel

        >>> processor = AutoProcessor.from_pretrained("google/pix2struct-textcaps-base")
        >>> model = Pix2StructTextModel.from_pretrained("google/pix2struct-textcaps-base")

        >>> inputs = processor(text="Hello, my dog is cute", return_tensors="pt")
        >>> outputs = model(**inputs)
        >>> loss = outputs.loss
        ```
        """
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError('You cannot specify both decoder_input_ids and decoder_inputs_embeds at the same time')
        elif input_ids is not None:
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError('You have to specify either decoder_input_ids or decoder_inputs_embeds')
        if inputs_embeds is None:
            assert self.embed_tokens is not None, 'You have to initialize the model with valid token embeddings'
            inputs_embeds = self.embed_tokens(input_ids)
        batch_size, seq_length = input_shape
        mask_seq_length = past_key_values[0][0].shape[2] + seq_length if past_key_values is not None else seq_length
        if attention_mask is None:
            attention_mask = torch.ones(batch_size, mask_seq_length, device=inputs_embeds.device)
        if encoder_attention_mask is None and encoder_hidden_states is not None:
            encoder_seq_length = encoder_hidden_states.shape[1]
            encoder_attention_mask = torch.ones(batch_size, encoder_seq_length, device=inputs_embeds.device, dtype=torch.long)
        if past_key_values is None:
            past_key_values = [None] * len(self.layer)
        extended_attention_mask = self.get_extended_attention_mask(attention_mask, input_shape)
        if encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=inputs_embeds.device)
            encoder_extended_attention_mask = self.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_extended_attention_mask = None
        head_mask = self.get_head_mask(head_mask, self.config.num_layers)
        cross_attn_head_mask = self.get_head_mask(cross_attn_head_mask, self.config.num_layers)
        present_key_value_states = () if use_cache else None
        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None
        all_cross_attentions = () if output_attentions else None
        position_bias = None
        encoder_decoder_position_bias = None
        hidden_states = self.dropout(inputs_embeds)
        for i, (layer_module, past_key_value) in enumerate(zip(self.layer, past_key_values)):
            layer_head_mask = head_mask[i]
            cross_attn_layer_head_mask = cross_attn_head_mask[i]
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)
            if self.gradient_checkpointing and self.training:
                if use_cache:
                    logger.warning('`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`...')
                    use_cache = False
                layer_outputs = self._gradient_checkpointing_func(layer_module.forward, hidden_states, extended_attention_mask, position_bias, encoder_hidden_states, encoder_extended_attention_mask, encoder_decoder_position_bias, layer_head_mask, cross_attn_layer_head_mask, None, use_cache, output_attentions)
            else:
                layer_outputs = layer_module(hidden_states, attention_mask=extended_attention_mask, position_bias=position_bias, encoder_hidden_states=encoder_hidden_states, encoder_attention_mask=encoder_extended_attention_mask, encoder_decoder_position_bias=encoder_decoder_position_bias, layer_head_mask=layer_head_mask, cross_attn_layer_head_mask=cross_attn_layer_head_mask, past_key_value=past_key_value, use_cache=use_cache, output_attentions=output_attentions)
            if use_cache is False:
                layer_outputs = layer_outputs[:1] + (None,) + layer_outputs[1:]
            hidden_states, present_key_value_state = layer_outputs[:2]
            position_bias = layer_outputs[2]
            if encoder_hidden_states is not None:
                encoder_decoder_position_bias = layer_outputs[4 if output_attentions else 3]
            if use_cache:
                present_key_value_states = present_key_value_states + (present_key_value_state,)
            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[3],)
                if encoder_hidden_states is not None:
                    all_cross_attentions = all_cross_attentions + (layer_outputs[5],)
        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = self.dropout(hidden_states)
        logits = self.lm_head(hidden_states)
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)
        loss = None
        if labels is not None:
            labels = labels.to(logits.device)
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100, reduction='mean')
            loss = loss_fct(logits.contiguous().view(-1, logits.size(-1)), labels.contiguous().view(-1))
        if not return_dict:
            return tuple((v for v in [loss, logits, present_key_value_states, all_hidden_states, all_attentions, all_cross_attentions] if v is not None))
        return CausalLMOutputWithCrossAttentions(loss=loss, logits=logits, past_key_values=present_key_value_states, hidden_states=all_hidden_states, attentions=all_attentions, cross_attentions=all_cross_attentions)