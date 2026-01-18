import math
import os
from dataclasses import dataclass
from typing import Optional, Tuple, Union
import numpy as np
import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from ...activations import ACT2FN
from ...modeling_outputs import (
from ...modeling_utils import PreTrainedModel
from ...pytorch_utils import apply_chunking_to_forward
from ...utils import (
from .configuration_big_bird import BigBirdConfig
@add_start_docstrings('The bare BigBird Model transformer outputting raw hidden-states without any specific head on top.', BIG_BIRD_START_DOCSTRING)
class BigBirdModel(BigBirdPreTrainedModel):
    """

    The model can behave as an encoder (with only self-attention) as well as a decoder, in which case a layer of
    cross-attention is added between the self-attention layers, following the architecture described in [Attention is
    all you need](https://arxiv.org/abs/1706.03762) by Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit,
    Llion Jones, Aidan N. Gomez, Lukasz Kaiser and Illia Polosukhin.

    To behave as an decoder the model needs to be initialized with the `is_decoder` argument of the configuration set
    to `True`. To be used in a Seq2Seq model, the model needs to initialized with both `is_decoder` argument and
    `add_cross_attention` set to `True`; an `encoder_hidden_states` is then expected as an input to the forward pass.
    """

    def __init__(self, config, add_pooling_layer=True):
        super().__init__(config)
        self.attention_type = self.config.attention_type
        self.config = config
        self.block_size = self.config.block_size
        self.embeddings = BigBirdEmbeddings(config)
        self.encoder = BigBirdEncoder(config)
        if add_pooling_layer:
            self.pooler = nn.Linear(config.hidden_size, config.hidden_size)
            self.activation = nn.Tanh()
        else:
            self.pooler = None
            self.activation = None
        if self.attention_type != 'original_full' and config.add_cross_attention:
            logger.warning('When using `BigBirdForCausalLM` as decoder, then `attention_type` must be `original_full`. Setting `attention_type=original_full`')
            self.set_attention_type('original_full')
        self.post_init()

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

    def set_attention_type(self, value: str):
        if value not in ['original_full', 'block_sparse']:
            raise ValueError(f"attention_type can only be set to either 'original_full' or 'block_sparse', but is {value}")
        if value == self.attention_type:
            return
        self.attention_type = value
        self.encoder.set_attention_type(value)

    @add_start_docstrings_to_model_forward(BIG_BIRD_INPUTS_DOCSTRING.format('batch_size, sequence_length'))
    @add_code_sample_docstrings(checkpoint=_CHECKPOINT_FOR_DOC, output_type=BaseModelOutputWithPoolingAndCrossAttentions, config_class=_CONFIG_FOR_DOC)
    def forward(self, input_ids: torch.LongTensor=None, attention_mask: Optional[torch.FloatTensor]=None, token_type_ids: Optional[torch.LongTensor]=None, position_ids: Optional[torch.LongTensor]=None, head_mask: Optional[torch.FloatTensor]=None, inputs_embeds: Optional[torch.FloatTensor]=None, encoder_hidden_states: Optional[torch.FloatTensor]=None, encoder_attention_mask: Optional[torch.FloatTensor]=None, past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]]=None, use_cache: Optional[bool]=None, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None) -> Union[BaseModelOutputWithPoolingAndCrossAttentions, Tuple[torch.FloatTensor]]:
        """
        encoder_hidden_states  (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention if
            the model is configured as a decoder.
        encoder_attention_mask (`torch.FloatTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on the padding token indices of the encoder input. This mask is used in
            the cross-attention if the model is configured as a decoder. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.
        past_key_values (`tuple(tuple(torch.FloatTensor))` of length `config.n_layers` with each tuple having 4 tensors of shape `(batch_size, num_heads, sequence_length - 1, embed_size_per_head)`):
            Contains precomputed key and value hidden states of the attention blocks. Can be used to speed up decoding.
            If `past_key_values` are used, the user can optionally input only the last `decoder_input_ids` (those that
            don't have their past key value states given to this model) of shape `(batch_size, 1)` instead of all
            `decoder_input_ids` of shape `(batch_size, sequence_length)`.
        use_cache (`bool`, *optional*):
            If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
            `past_key_values`).
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        if self.config.is_decoder:
            use_cache = use_cache if use_cache is not None else self.config.use_cache
        else:
            use_cache = False
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError('You cannot specify both input_ids and inputs_embeds at the same time')
        elif input_ids is not None:
            self.warn_if_padding_and_no_attention_mask(input_ids, attention_mask)
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError('You have to specify either input_ids or inputs_embeds')
        batch_size, seq_length = input_shape
        device = input_ids.device if input_ids is not None else inputs_embeds.device
        past_key_values_length = past_key_values[0][0].shape[2] if past_key_values is not None else 0
        if attention_mask is None:
            attention_mask = torch.ones((batch_size, seq_length + past_key_values_length), device=device)
        if token_type_ids is None:
            if hasattr(self.embeddings, 'token_type_ids'):
                buffered_token_type_ids = self.embeddings.token_type_ids[:, :seq_length]
                buffered_token_type_ids_expanded = buffered_token_type_ids.expand(batch_size, seq_length)
                token_type_ids = buffered_token_type_ids_expanded
            else:
                token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)
        max_tokens_to_attend = (5 + 2 * self.config.num_random_blocks) * self.config.block_size
        if self.attention_type == 'block_sparse' and seq_length <= max_tokens_to_attend:
            sequence_length = input_ids.size(1) if input_ids is not None else inputs_embeds.size(1)
            logger.warning(f"Attention type 'block_sparse' is not possible if sequence_length: {sequence_length} <= num global tokens: 2 * config.block_size + min. num sliding tokens: 3 * config.block_size + config.num_random_blocks * config.block_size + additional buffer: config.num_random_blocks * config.block_size = {max_tokens_to_attend} with config.block_size = {self.config.block_size}, config.num_random_blocks = {self.config.num_random_blocks}. Changing attention type to 'original_full'...")
            self.set_attention_type('original_full')
        if self.attention_type == 'block_sparse':
            padding_len, input_ids, attention_mask, token_type_ids, position_ids, inputs_embeds = self._pad_to_block_size(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, position_ids=position_ids, inputs_embeds=inputs_embeds, pad_token_id=self.config.pad_token_id)
        else:
            padding_len = 0
        if self.attention_type == 'block_sparse':
            blocked_encoder_mask, band_mask, from_mask, to_mask = self.create_masks_for_block_sparse_attn(attention_mask, self.block_size)
            extended_attention_mask = None
        elif self.attention_type == 'original_full':
            blocked_encoder_mask = None
            band_mask = None
            from_mask = None
            to_mask = None
            extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(attention_mask, input_shape)
        else:
            raise ValueError(f'attention_type can either be original_full or block_sparse, but is {self.attention_type}')
        if self.config.is_decoder and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)
            encoder_extended_attention_mask = self.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_extended_attention_mask = None
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)
        embedding_output = self.embeddings(input_ids=input_ids, position_ids=position_ids, token_type_ids=token_type_ids, inputs_embeds=inputs_embeds, past_key_values_length=past_key_values_length)
        encoder_outputs = self.encoder(embedding_output, attention_mask=extended_attention_mask, head_mask=head_mask, encoder_hidden_states=encoder_hidden_states, encoder_attention_mask=encoder_extended_attention_mask, past_key_values=past_key_values, use_cache=use_cache, output_attentions=output_attentions, output_hidden_states=output_hidden_states, band_mask=band_mask, from_mask=from_mask, to_mask=to_mask, blocked_encoder_mask=blocked_encoder_mask, return_dict=return_dict)
        sequence_output = encoder_outputs[0]
        pooler_output = self.activation(self.pooler(sequence_output[:, 0, :])) if self.pooler is not None else None
        if padding_len > 0:
            sequence_output = sequence_output[:, :-padding_len]
        if not return_dict:
            return (sequence_output, pooler_output) + encoder_outputs[1:]
        return BaseModelOutputWithPoolingAndCrossAttentions(last_hidden_state=sequence_output, pooler_output=pooler_output, past_key_values=encoder_outputs.past_key_values, hidden_states=encoder_outputs.hidden_states, attentions=encoder_outputs.attentions, cross_attentions=encoder_outputs.cross_attentions)

    @staticmethod
    def create_masks_for_block_sparse_attn(attention_mask: torch.Tensor, block_size: int):
        batch_size, seq_length = attention_mask.size()
        if seq_length % block_size != 0:
            raise ValueError(f'Sequence length must be multiple of block size, but sequence length is {seq_length}, while block size is {block_size}.')

        def create_band_mask_from_inputs(from_blocked_mask, to_blocked_mask):
            """
            Create 3D attention mask from a 2D tensor mask.

            Args:
                from_blocked_mask: 2D Tensor of shape [batch_size,
                from_seq_length//from_block_size, from_block_size].
                to_blocked_mask: int32 Tensor of shape [batch_size,
                to_seq_length//to_block_size, to_block_size].

            Returns:
                float Tensor of shape [batch_size, 1, from_seq_length//from_block_size-4, from_block_size,
                3*to_block_size].
            """
            exp_blocked_to_pad = torch.cat([to_blocked_mask[:, 1:-3], to_blocked_mask[:, 2:-2], to_blocked_mask[:, 3:-1]], dim=2)
            band_mask = torch.einsum('blq,blk->blqk', from_blocked_mask[:, 2:-2], exp_blocked_to_pad)
            band_mask.unsqueeze_(1)
            return band_mask
        blocked_encoder_mask = attention_mask.view(batch_size, seq_length // block_size, block_size)
        band_mask = create_band_mask_from_inputs(blocked_encoder_mask, blocked_encoder_mask)
        from_mask = attention_mask.view(batch_size, 1, seq_length, 1)
        to_mask = attention_mask.view(batch_size, 1, 1, seq_length)
        return (blocked_encoder_mask, band_mask, from_mask, to_mask)

    def _pad_to_block_size(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, token_type_ids: torch.Tensor, position_ids: torch.Tensor, inputs_embeds: torch.Tensor, pad_token_id: int):
        """A helper function to pad tokens and mask to work with implementation of BigBird block-sparse attention."""
        block_size = self.config.block_size
        input_shape = input_ids.shape if input_ids is not None else inputs_embeds.shape
        batch_size, seq_len = input_shape[:2]
        padding_len = (block_size - seq_len % block_size) % block_size
        if padding_len > 0:
            logger.warning_once(f'Input ids are automatically padded from {seq_len} to {seq_len + padding_len} to be a multiple of `config.block_size`: {block_size}')
            if input_ids is not None:
                input_ids = nn.functional.pad(input_ids, (0, padding_len), value=pad_token_id)
            if position_ids is not None:
                position_ids = nn.functional.pad(position_ids, (0, padding_len), value=pad_token_id)
            if inputs_embeds is not None:
                input_ids_padding = inputs_embeds.new_full((batch_size, padding_len), self.config.pad_token_id, dtype=torch.long)
                inputs_embeds_padding = self.embeddings(input_ids_padding)
                inputs_embeds = torch.cat([inputs_embeds, inputs_embeds_padding], dim=-2)
            attention_mask = nn.functional.pad(attention_mask, (0, padding_len), value=False)
            token_type_ids = nn.functional.pad(token_type_ids, (0, padding_len), value=0)
        return (padding_len, input_ids, attention_mask, token_type_ids, position_ids, inputs_embeds)