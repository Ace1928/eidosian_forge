import math
import os
import warnings
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Union
import torch
from torch import nn
from torch.nn import CrossEntropyLoss, SmoothL1Loss
from ...activations import ACT2FN, gelu
from ...modeling_utils import PreTrainedModel
from ...utils import (
from .configuration_lxmert import LxmertConfig
@add_start_docstrings('The bare Lxmert Model transformer outputting raw hidden-states without any specific head on top.', LXMERT_START_DOCSTRING)
class LxmertModel(LxmertPreTrainedModel):

    def __init__(self, config):
        super().__init__(config)
        self.embeddings = LxmertEmbeddings(config)
        self.encoder = LxmertEncoder(config)
        self.pooler = LxmertPooler(config)
        self.post_init()

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, new_embeddings):
        self.embeddings.word_embeddings = new_embeddings

    @add_start_docstrings_to_model_forward(LXMERT_INPUTS_DOCSTRING.format('batch_size, sequence_length'))
    @add_code_sample_docstrings(checkpoint=_CHECKPOINT_FOR_DOC, output_type=LxmertModelOutput, config_class=_CONFIG_FOR_DOC)
    def forward(self, input_ids: Optional[torch.LongTensor]=None, visual_feats: Optional[torch.FloatTensor]=None, visual_pos: Optional[torch.FloatTensor]=None, attention_mask: Optional[torch.FloatTensor]=None, visual_attention_mask: Optional[torch.FloatTensor]=None, token_type_ids: Optional[torch.LongTensor]=None, inputs_embeds: Optional[torch.FloatTensor]=None, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None) -> Union[LxmertModelOutput, Tuple[torch.FloatTensor]]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError('You cannot specify both input_ids and inputs_embeds at the same time')
        elif input_ids is not None:
            self.warn_if_padding_and_no_attention_mask(input_ids, attention_mask)
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError('You have to specify either input_ids or inputs_embeds')
        if visual_feats is None:
            raise ValueError('`visual_feats` cannot be `None`')
        if visual_pos is None:
            raise ValueError('`visual_pos` cannot be `None`')
        device = input_ids.device if input_ids is not None else inputs_embeds.device
        if attention_mask is None:
            attention_mask = torch.ones(input_shape, device=device)
        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = extended_attention_mask.to(dtype=self.dtype)
        extended_attention_mask = (1.0 - extended_attention_mask) * torch.finfo(self.dtype).min
        if visual_attention_mask is not None:
            extended_visual_attention_mask = visual_attention_mask.unsqueeze(1).unsqueeze(2)
            extended_visual_attention_mask = extended_visual_attention_mask.to(dtype=self.dtype)
            extended_visual_attention_mask = (1.0 - extended_visual_attention_mask) * torch.finfo(self.dtype).min
        else:
            extended_visual_attention_mask = None
        embedding_output = self.embeddings(input_ids, token_type_ids, inputs_embeds)
        encoder_outputs = self.encoder(embedding_output, extended_attention_mask, visual_feats=visual_feats, visual_pos=visual_pos, visual_attention_mask=extended_visual_attention_mask, output_attentions=output_attentions)
        visual_encoder_outputs, lang_encoder_outputs = encoder_outputs[:2]
        vision_hidden_states = visual_encoder_outputs[0]
        language_hidden_states = lang_encoder_outputs[0]
        all_attentions = ()
        if output_attentions:
            language_attentions = lang_encoder_outputs[1]
            vision_attentions = visual_encoder_outputs[1]
            cross_encoder_attentions = encoder_outputs[2]
            all_attentions = (language_attentions, vision_attentions, cross_encoder_attentions)
        hidden_states = (language_hidden_states, vision_hidden_states) if output_hidden_states else ()
        visual_output = vision_hidden_states[-1]
        lang_output = language_hidden_states[-1]
        pooled_output = self.pooler(lang_output)
        if not return_dict:
            return (lang_output, visual_output, pooled_output) + hidden_states + all_attentions
        return LxmertModelOutput(pooled_output=pooled_output, language_output=lang_output, vision_output=visual_output, language_hidden_states=language_hidden_states if output_hidden_states else None, vision_hidden_states=vision_hidden_states if output_hidden_states else None, language_attentions=language_attentions if output_attentions else None, vision_attentions=vision_attentions if output_attentions else None, cross_encoder_attentions=cross_encoder_attentions if output_attentions else None)