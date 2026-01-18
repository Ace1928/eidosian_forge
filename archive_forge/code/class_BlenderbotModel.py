import copy
import math
import os
import warnings
from typing import List, Optional, Tuple, Union
import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import CrossEntropyLoss
from ...activations import ACT2FN
from ...modeling_attn_mask_utils import _prepare_4d_attention_mask, _prepare_4d_causal_attention_mask
from ...modeling_outputs import (
from ...modeling_utils import PreTrainedModel
from ...utils import (
from ..blenderbot_small import BlenderbotSmallForConditionalGeneration, BlenderbotSmallModel
from .configuration_blenderbot import BlenderbotConfig
@add_start_docstrings('The bare Blenderbot Model outputting raw hidden-states without any specific head on top.', BLENDERBOT_START_DOCSTRING)
class BlenderbotModel(BlenderbotPreTrainedModel):
    _tied_weights_keys = ['decoder.embed_tokens.weight', 'encoder.embed_tokens.weight']

    def __init__(self, config: BlenderbotConfig):
        super().__init__(config)
        padding_idx, vocab_size = (config.pad_token_id, config.vocab_size)
        self.shared = nn.Embedding(vocab_size, config.d_model, padding_idx)
        self.encoder = BlenderbotEncoder(config, self.shared)
        self.decoder = BlenderbotDecoder(config, self.shared)
        self.post_init()

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: Optional[Union[str, os.PathLike]], *model_args, **kwargs):
        if pretrained_model_name_or_path == 'facebook/blenderbot-90M':
            warnings.warn("The checkpoint `facebook/blenderbot-90M` is deprecated. In the future, please use the identical checkpoint `facebook/small_blenderbot-90M` with `BlenderbotSmallModel.from_pretrained('facebook/small_blenderbot-90M')` instead.", FutureWarning)
            return BlenderbotSmallModel.from_pretrained(pretrained_model_name_or_path)
        return super(BlenderbotModel, cls).from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)

    def get_input_embeddings(self):
        return self.shared

    def set_input_embeddings(self, value):
        self.shared = value
        self.encoder.embed_tokens = self.shared
        self.decoder.embed_tokens = self.shared

    def get_encoder(self):
        return self.encoder

    def get_decoder(self):
        return self.decoder

    @add_start_docstrings_to_model_forward(BLENDERBOT_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=Seq2SeqModelOutput, config_class=_CONFIG_FOR_DOC)
    def forward(self, input_ids: Optional[torch.LongTensor]=None, attention_mask: Optional[torch.Tensor]=None, decoder_input_ids: Optional[torch.LongTensor]=None, decoder_attention_mask: Optional[torch.LongTensor]=None, head_mask: Optional[torch.Tensor]=None, decoder_head_mask: Optional[torch.Tensor]=None, cross_attn_head_mask: Optional[torch.Tensor]=None, encoder_outputs: Optional[Union[Tuple, BaseModelOutput]]=None, past_key_values: Optional[List[torch.FloatTensor]]=None, inputs_embeds: Optional[torch.Tensor]=None, decoder_inputs_embeds: Optional[torch.FloatTensor]=None, use_cache: Optional[bool]=None, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None) -> Union[Tuple[torch.FloatTensor], Seq2SeqModelOutput]:
        """
        Returns:

        Example:

        ```python
        >>> from transformers import AutoTokenizer, BlenderbotModel

        >>> model = BlenderbotModel.from_pretrained("facebook/blenderbot-400M-distill")
        >>> tokenizer = AutoTokenizer.from_pretrained("facebook/blenderbot-400M-distill")

        >>> inputs = tokenizer("Studies have been shown that owning a dog is good for you", return_tensors="pt")
        >>> decoder_input_ids = tokenizer("Studies show that", return_tensors="pt").input_ids  # Batch size 1
        >>> outputs = model(input_ids=inputs.input_ids, decoder_input_ids=decoder_input_ids)

        >>> last_hidden_states = outputs.last_hidden_state
        >>> list(last_hidden_states.shape)
        [1, 6, 1280]
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