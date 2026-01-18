import copy
import math
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import CrossEntropyLoss
from ...activations import ACT2FN
from ...modeling_attn_mask_utils import _prepare_4d_attention_mask, _prepare_4d_causal_attention_mask
from ...modeling_outputs import (
from ...modeling_utils import PreTrainedModel
from ...utils import (
from .configuration_marian import MarianConfig
@add_start_docstrings('The bare Marian Model outputting raw hidden-states without any specific head on top.', MARIAN_START_DOCSTRING)
class MarianModel(MarianPreTrainedModel):
    _tied_weights_keys = ['encoder.embed_tokens.weight', 'decoder.embed_tokens.weight']

    def __init__(self, config: MarianConfig):
        super().__init__(config)
        padding_idx, vocab_size = (config.pad_token_id, config.vocab_size)
        self.shared = nn.Embedding(vocab_size, config.d_model, padding_idx)
        if self.config.share_encoder_decoder_embeddings:
            encoder_embed_tokens = decoder_embed_tokens = self.shared
        else:
            encoder_embed_tokens = copy.deepcopy(self.shared)
            decoder_embed_tokens = copy.deepcopy(self.shared)
            self.shared = None
        self.encoder = MarianEncoder(config, encoder_embed_tokens)
        self.decoder = MarianDecoder(config, decoder_embed_tokens)
        self.post_init()

    def get_input_embeddings(self):
        return self.get_encoder().get_input_embeddings()

    def set_input_embeddings(self, value):
        if self.config.share_encoder_decoder_embeddings:
            self.shared = value
            self.encoder.embed_tokens = self.shared
            self.decoder.embed_tokens = self.shared
        else:
            self.encoder.embed_tokens = value

    def get_decoder_input_embeddings(self):
        if self.config.share_encoder_decoder_embeddings:
            raise ValueError('`get_decoder_input_embeddings` should not be called if `config.share_encoder_decoder_embeddings` is `True`. Please use `get_input_embeddings` instead.')
        return self.get_decoder().get_input_embeddings()

    def set_decoder_input_embeddings(self, value):
        if self.config.share_encoder_decoder_embeddings:
            raise ValueError('`config.share_encoder_decoder_embeddings` is set to `True` meaning the decoder input embeddings are shared with the encoder. In order to set the decoder input embeddings, you should simply set the encoder input embeddings by calling `set_input_embeddings` with the appropriate embeddings.')
        self.decoder.embed_tokens = value

    def get_encoder(self):
        return self.encoder

    def get_decoder(self):
        return self.decoder

    def resize_decoder_token_embeddings(self, new_num_tokens: int) -> nn.Embedding:
        if self.config.share_encoder_decoder_embeddings:
            raise ValueError('`resize_decoder_token_embeddings` should not be called if `config.share_encoder_decoder_embeddings` is `True`. Please use `resize_token_embeddings` instead.')
        old_embeddings = self.get_decoder_input_embeddings()
        new_embeddings = self._get_resized_embeddings(old_embeddings, new_num_tokens)
        self.set_decoder_input_embeddings(new_embeddings)
        model_embeds = self.get_decoder_input_embeddings()
        if new_num_tokens is None:
            return model_embeds
        self.config.decoder_vocab_size = new_num_tokens
        self.tie_weights()
        return model_embeds

    @add_start_docstrings_to_model_forward(MARIAN_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=Seq2SeqModelOutput, config_class=_CONFIG_FOR_DOC)
    def forward(self, input_ids: torch.LongTensor=None, attention_mask: Optional[torch.Tensor]=None, decoder_input_ids: Optional[torch.LongTensor]=None, decoder_attention_mask: Optional[torch.Tensor]=None, head_mask: Optional[torch.Tensor]=None, decoder_head_mask: Optional[torch.Tensor]=None, cross_attn_head_mask: Optional[torch.Tensor]=None, encoder_outputs: Optional[Union[Tuple[torch.Tensor], BaseModelOutput]]=None, past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]]=None, inputs_embeds: Optional[torch.FloatTensor]=None, decoder_inputs_embeds: Optional[torch.FloatTensor]=None, use_cache: Optional[bool]=None, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None) -> Seq2SeqModelOutput:
        """
        Returns:

        Example:

        ```python
        >>> from transformers import AutoTokenizer, MarianModel

        >>> tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-de")
        >>> model = MarianModel.from_pretrained("Helsinki-NLP/opus-mt-en-de")

        >>> inputs = tokenizer("Studies have been shown that owning a dog is good for you", return_tensors="pt")
        >>> decoder_inputs = tokenizer(
        ...     "<pad> Studien haben gezeigt dass es hilfreich ist einen Hund zu besitzen",
        ...     return_tensors="pt",
        ...     add_special_tokens=False,
        ... )
        >>> outputs = model(input_ids=inputs.input_ids, decoder_input_ids=decoder_inputs.input_ids)

        >>> last_hidden_states = outputs.last_hidden_state
        >>> list(last_hidden_states.shape)
        [1, 26, 512]
        ```"""
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        if encoder_outputs is None:
            encoder_outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask, head_mask=head_mask, inputs_embeds=inputs_embeds, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict)
        elif return_dict and (not isinstance(encoder_outputs, BaseModelOutput)):
            encoder_outputs = BaseModelOutput(last_hidden_state=encoder_outputs[0], hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None, attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None)
        decoder_outputs = self.decoder(input_ids=decoder_input_ids, attention_mask=decoder_attention_mask, encoder_hidden_states=encoder_outputs[0], encoder_attention_mask=attention_mask, head_mask=decoder_head_mask, cross_attn_head_mask=cross_attn_head_mask, past_key_values=past_key_values, inputs_embeds=decoder_inputs_embeds, use_cache=use_cache, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict)
        if not return_dict:
            return decoder_outputs + encoder_outputs
        return Seq2SeqModelOutput(last_hidden_state=decoder_outputs.last_hidden_state, past_key_values=decoder_outputs.past_key_values, decoder_hidden_states=decoder_outputs.hidden_states, decoder_attentions=decoder_outputs.attentions, cross_attentions=decoder_outputs.cross_attentions, encoder_last_hidden_state=encoder_outputs.last_hidden_state, encoder_hidden_states=encoder_outputs.hidden_states, encoder_attentions=encoder_outputs.attentions)